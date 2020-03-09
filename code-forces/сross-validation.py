def main():
    first_line = [int(item) for item in input().split(' ')]
    # objects count
    n = first_line[0]
    # classes count
    m = first_line[1]
    # parts count
    k = first_line[2]

    objects = [int(item) for item in input().split(' ')]

    objects_by_class = []
    for i in range(m):
        objects_by_class.append([])
    for i in range(n):
        objects_by_class[objects[i] - 1].append(i)

    objects_by_part = []

    for i in range(k):
        objects_by_part.append([])

    j = -1
    for some_class in objects_by_class:
        for i in range(len(some_class)):
            if j == k - 1:
                j = 0
            else:
                j += 1
            objects_by_part[j].append(some_class[i])

    for some_part in objects_by_part:
        print(len(some_part), end='')
        sorted_object_by_part = sorted(some_part)
        for number in sorted_object_by_part:
            print(' ', number + 1, end='')
        print()


if __name__ == "__main__":
    main()
