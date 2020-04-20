# Python 2.7.15
from io import BytesIO, IOBase
import os
import sys

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self, **kwargs):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip("\r\n")

x1_count, x2_count = input().split(' ')
x1_count = int(x1_count)
x2_count = int(x2_count)
n = int(input())

x1_counts = {}
x2_counts = {}
pairs_count = {}
for i in range(n):
    x1_temp, x2_temp = input().split(' ')
    x1_temp = int(x1_temp) - 1 # should be long
    x2_temp = int(x2_temp) - 1
    x1_counts.setdefault(x1_temp, 0)
    x1_counts[x1_temp] += 1
    x2_counts.setdefault(x2_temp, 0)
    x2_counts[x2_temp] += 1
    pairs_count.setdefault(x1_temp, {})
    pairs_count[x1_temp].setdefault(x2_temp, 0)
    pairs_count[x1_temp][x2_temp] += 1

hi2 = n
for key, value in pairs_count.items():
    for pkey, pvalue in value.items():
        val = x1_counts[key] * x2_counts[pkey] / float(n)
        hi2 += (pvalue - val) ** 2 / val - val

print(hi2)