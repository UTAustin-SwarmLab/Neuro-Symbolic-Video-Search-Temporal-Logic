"""test script of module."""

a = [1, 2]
b = [1, 3, 4]

result = all(item in b for item in a)
print(result)
