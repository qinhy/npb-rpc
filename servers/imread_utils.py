
from pathlib import Path
from typing import Literal
import warnings
import cv2
import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file


def read_jpeg_tensor(path: Path, device: torch.device | str) -> torch.Tensor:
    """Decode JPEG to CHW RGB uint8 directly on *device* when supported."""
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"input image must be JPEG: {path}")

    device = torch.device(device)
    encoded = read_file(str(path))
    if encoded.numel() == 0:
        raise ValueError(f"input JPEG is empty: {path}")

    try:
        image = decode_jpeg(
            encoded,
            mode=ImageReadMode.RGB,
            device=device,
        )
    except RuntimeError:
        if device.type != "cuda":
            raise
        # Some torchvision builds do not include CUDA/nvJPEG support.
        image = decode_jpeg(
            encoded,
            mode=ImageReadMode.RGB,
            device="cpu",
        ).to(device, non_blocking=True)

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected decoded RGB CHW image, got shape={tuple(image.shape)}")
    return image.contiguous()


def read_image(path, ops:"MatOps", color: Literal["RGB", "BGR", "gray"]="BGR"):        
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"input image must be JPEG: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"input JPEG is empty: {path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if color == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return ops.from_numpy(image)


try:

    import hashlib
    import threading
    from pathlib import Path
    from typing import TYPE_CHECKING, Any, Literal

    import numpy as np
    from turbojpeg import TJPF_BGR, TJPF_GRAY, TJPF_RGB, TurboJPEG

    import mstore
    from mstore import AuthenticationError, ObjectExists, ObjectNotFound, ProtocolError

    if TYPE_CHECKING:
        import torch

    Color = Literal["RGB", "BGR", "gray"]
    PIXEL_FORMATS: dict[str, int] = {"RGB": TJPF_RGB, "BGR": TJPF_BGR, "gray": TJPF_GRAY}
    CHANNELS: dict[int, int] = {TJPF_RGB: 3, TJPF_BGR: 3, TJPF_GRAY: 1}
    TJFLAG_FASTDCT = 2048

    TJ = TurboJPEG()
    MSTORE: mstore.Client | None = None
    _STORE_LOCK = threading.Lock()

    # Errors that mean "no usable server right now": the cache is optional.
    _UNAVAILABLE = (OSError, ProtocolError)


    def get_store() -> mstore.Client:
        """Process-wide client, created lazily and safe to share between threads."""
        global MSTORE
        with _STORE_LOCK:
            if MSTORE is None:
                MSTORE = mstore.connect()
            return MSTORE


    def cache_key(path: Path, color: Color) -> str:
        """Cache key for a decoded image; every process computes the same one.

        Built from path, size, mtime and colour, so an edited file or a different
        colour order never returns the wrong pixels. Used as the object id.
        """
        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{color}"
        digest = hashlib.blake2b(key.encode("utf-8", "surrogateescape"), digest_size=16).hexdigest()
        return f"jpeg-{digest}"


    def _pixel_format(color: str) -> int:
        try:
            return PIXEL_FORMATS[color]
        except KeyError:
            raise ValueError(f"color must be one of {sorted(PIXEL_FORMATS)}, got {color!r}") from None


    def _check_jpeg(path: Path) -> None:
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(f"input image must be JPEG: {path}")


    def decode_into(jpeg: bytes | np.ndarray, dst: np.ndarray, *, pixel_format: int = TJPF_RGB,
                    scaling_factor: tuple[int, int] | None = None, fast: bool = False) -> np.ndarray:
        """Decode ``jpeg`` into ``dst`` in place, refusing every case that would copy.

        PyTurboJPEG silently allocates a *new* array when dst does not match exactly,
        and it assumes a C-contiguous layout (pitch 0). Check all of that up front.
        """
        if not dst.flags.writeable:
            raise ValueError("dst is read-only: open the object with mode='write'")
        if not dst.flags.c_contiguous:
            raise ValueError("dst must be C-contiguous: create the object with order='C'")
        width, height, _sub, _cs = TJ.decode_header(jpeg)
        if scaling_factor:
            num, den = scaling_factor
            width, height = (width * num + den - 1) // den, (height * num + den - 1) // den
        expected = (height, width, CHANNELS[pixel_format])
        if dst.shape != expected or dst.dtype != np.uint8:
            raise ValueError(f"dst must be uint8 {expected}, got {dst.dtype} {dst.shape}")

        flags = TJFLAG_FASTDCT if fast else 0
        out = TJ.decode(jpeg, pixel_format=pixel_format, scaling_factor=scaling_factor,
                        flags=flags, dst=dst)
        if out is not dst:  # belt and braces against the silent-fallback behaviour
            raise RuntimeError("PyTurboJPEG decoded into a new array instead of dst")
        return out


    def _decode_new(path: Path, color: Color) -> np.ndarray:
        fmt = _pixel_format(color)
        jpeg = np.fromfile(path, dtype=np.uint8)
        width, height, _sub, _cs = TJ.decode_header(jpeg)
        out = np.empty((height, width, CHANNELS[fmt]), dtype=np.uint8)
        return decode_into(jpeg, out, pixel_format=fmt)


    # --------------------------------------------------------------------------- cache

    def set_cache(path: str | Path, image: np.ndarray, color: Color = "BGR") -> bool:
        """Put a decoded image in the store for the next read_image() of ``path``.

        ``image`` is uint8 HxWx3 (RGB/BGR) or HxW / HxWx1 (gray). It is copied, so the
        caller keeps its own array. Returns False if an entry already exists or the
        server is not running.
        """
        path = Path(path)
        fmt = _pixel_format(color)
        image = np.asarray(image)
        if color == "gray" and image.ndim == 2:
            image = image[..., None]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != CHANNELS[fmt]:
            raise ValueError(f"{color} image must be uint8 with {CHANNELS[fmt]} channel(s), "
                            f"got {image.dtype} {image.shape}")
        key = cache_key(path, color)
        try:
            obj = get_store().create(shape=image.shape, dtype=np.uint8, object_id=key,
                                    metadata={"format": color, "source": str(path)})
        except ObjectExists:
            return False  # already cached, or another caller is caching it right now
        except _UNAVAILABLE:
            return False
        try:
            view = obj.numpy()
            np.copyto(view, image)
            del view
            # Publish last. Until the entry is public nobody else can open it, so a
            # half-written image can never be popped.
            obj.make_public("admin")
            return True
        except _UNAVAILABLE:
            return False
        except BaseException:
            try:
                obj.delete()  # don't leave a broken entry behind
            except Exception:
                pass
            raise
        finally:
            obj.close()


    def pop_cache(path: str | Path, color: Color = "BGR") -> np.ndarray | None:
        """Take the cached image for ``path`` out of the store, or return None.

        Exactly one caller gets each entry, even when several threads or processes
        pop at the same moment: the server-side delete decides the winner.
        """
        path = Path(path)
        _pixel_format(color)
        key = cache_key(path, color)
        try:
            obj = get_store().open(key, mode="write")  # public entry: no token
        except (ObjectNotFound, AuthenticationError):
            return None  # not cached, or still being written by another caller
        except _UNAVAILABLE:
            return None
        try:
            obj.delete()  # atomic on the server: only one popper succeeds
            return obj.numpy()  # the mapping outlives the delete; freed with the array
        except (ObjectNotFound, AuthenticationError):
            # Another caller popped it first. AuthenticationError: it was popped and a
            # third caller is already writing a new, not yet published entry for this key.
            return None
        except _UNAVAILABLE:
            return None
        finally:
            obj.close()


    # --------------------------------------------------------------------------- readers

    def read_image(path: str | Path, ops: "MatOps | None" = None, color: Color = "BGR",
                *, cache_on_miss: bool = True) -> np.ndarray:
        """Read a JPEG as uint8 HxWx3 (RGB/BGR) or HxW (gray).

        Pops the image from the shared cache when it is there; otherwise decodes the
        file and, with ``cache_on_miss=True``, leaves a copy in the cache for the next
        caller (any thread, any process). The result is always writable and private.
        """
        path = Path(path)
        _check_jpeg(path)
        image = pop_cache(path, color)        
        # print("[read_image]",image is None , cache_on_miss , "hand" in str(path), "rgb.jpg" in str(path))
        if image is None:
            image = _decode_new(path, color)
            if cache_on_miss and "hand" in str(path) and "rgb.jpg" in str(path):
                set_cache(path, image, color)
        if color == "gray":
            image = image[..., 0]
        # If MatOps needs its own type, convert here, e.g. return ops.<from_numpy>(image)
        return image


    def read_jpeg_tensor(path: str | Path, device: "torch.device | str" = "cpu", *,
                        color: Color = "RGB", cache_on_miss: bool = True) -> "torch.Tensor":
        """read_image() as a uint8 tensor of shape (C, H, W) on ``device``."""
        import torch

        image = read_image(path, None, color, cache_on_miss=cache_on_miss)
        if image.ndim == 2:
            image = image[..., None]
        return torch.from_numpy(image).permute(2, 0, 1).to(device)
    
except Exception as e:
    print("no store jpeg support",e)