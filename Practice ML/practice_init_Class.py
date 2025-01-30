class MyClass:
    pass

o1 = MyClass()
o2 = MyClass()
print(type(o1))

o1.x = 10
o1.y = 20
print(vars(o1))
o2.a = {1, 2, 3, 4}
o2.b = {'a':1, 'b': 2, 'c':3}
print(vars(o2))

class MyClass:

    def __init__(self, x, y):
        print(f'Now running MyClass.__init__, with {x = } and {y = }')
        self.x = x
        self.y = y

o1 = MyClass(10,20)
print(vars(o1))