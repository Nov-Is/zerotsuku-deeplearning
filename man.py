class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        self.myoji = "higuma"
        print("Good-bye " + self.myoji + " " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()
