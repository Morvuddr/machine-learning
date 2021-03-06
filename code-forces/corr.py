# Python 2.7.15
from io import BytesIO, IOBase
import os
import sys
from math import sqrt

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

n = int(input())
entities = []
for i in range(n):
    x1, x2 = input().split(' ')
    x1 = int(x1)
    x2 = int(x2)
    entities.append((x1, x2))

average_x1 = sum([x[0] for x in entities]) / float(len([x[0] for x in entities]))
average_x2 = sum([x[1] for x in entities]) / float(len([x[1] for x in entities]))

top = 0.0
dispersion_x1 = 0.0
dispersion_x2 = 0.0

for e in entities:
    diff_x1 = e[0] - average_x1
    diff_x2 = e[1] - average_x2
    top += diff_x1 * diff_x2
    dispersion_x1 += diff_x1 ** 2
    dispersion_x2 += diff_x2 ** 2

bot = sqrt(dispersion_x1 * dispersion_x2)
result = 0.0
if bot == 0:
    result = 0
else:
    result = top / bot
print(result)