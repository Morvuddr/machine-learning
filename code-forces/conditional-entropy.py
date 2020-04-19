# Python 2.7.15

from io import BytesIO, IOBase
import os
import sys
from math import log

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


kx, ky = input().split(' ')
kx = int(kx)
ky = int(ky)
n = int(input())

y_objects = [0] * n
x_counts = [0] * kx
pairs_count = [0] * ky
indexes = {a: [] for a in range(kx)}
index = set()
for i in range(0, n, 1):
    line = input().split(' ')
    x = int(line[0]) - 1
    y = int(line[1]) - 1
    x_counts[x] += 1
    y_objects[i] = y
    temp = indexes[x]
    temp.append(i)
    indexes[x] = temp

entropy = 0.0
cond_prod = 0.0
for i in range(0, kx, 1):
    if x_counts[i] != 0:
        j = len(indexes[i])
        for k in indexes[i]:
            value_y = y_objects[k]
            pairs_count[value_y] += 1
            index.add(value_y)
        for k in index:
            value = pairs_count[k] / float(j)
            if value == 0:
                break
            cond_prod += value * log(value)
            pairs_count[k] = 0
        index.clear()
        entropy += cond_prod * j / n
        cond_prod = 0.0

print(-entropy)