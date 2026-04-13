from typing import Iterable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def progress_bar(iterable: Iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    return SimpleProgress(iterable, **kwargs)


class SimpleProgress:
    def __init__(self, iterable: Iterable, desc=None, total=None, leave=True):
        self.iterable = iterable
        self.desc = desc or "Progress"
        self.total = total
        self.leave = leave
        self.count = 0
        self._last_printed = -1

    def __iter__(self):
        if self.total:
            print(f"{self.desc}: 0/{self.total}")
        else:
            print(f"{self.desc}: started")

        for item in self.iterable:
            self.count += 1
            self._maybe_print()
            yield item

        if self.leave:
            if self.total:
                print(f"{self.desc}: {self.total}/{self.total} complete")
            else:
                print(f"{self.desc}: complete ({self.count} items)")

    def _maybe_print(self):
        if self.total:
            percent = int((100 * self.count) / self.total)
            if percent >= self._last_printed + 10 or self.count == self.total:
                self._last_printed = percent
                print(f"{self.desc}: {self.count}/{self.total} ({percent}%)")
        elif self.count % 100 == 0:
            print(f"{self.desc}: processed {self.count}")

    def set_postfix(self, **kwargs):
        return None

    def close(self):
        return None
