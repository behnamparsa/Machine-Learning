class Animal:
    def __init__(self, name, num_legs):
        self.name = name
        self.num_legs = num_legs
    
    def classify(self):
        if self.num_legs == 0:
            return "I am a snake!"
        elif self.num_legs == 4:
            return "I am a dog!"
        elif self.num_legs == 2:
            return "I am a human!"
        else:
            return "I am an unknown species!"

    def sound(self):
      print("do nothing")

class Dog(Animal):
    def __init__(self, name, breed, fur_color):
        super().__init__(name, 4)
        self.breed = breed
        self.fur_color = fur_color

    def sound(self):
        print('just Bark')

dog = Dog("Fido", "Labrador", "brown")

print(dog.classify())
print(dog.sound())
