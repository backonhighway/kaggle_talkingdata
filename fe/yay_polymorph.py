class Task:
    def __init__(self):
        print("initialized")

    def doit(self):
        print("omg")


class SubTaskA(Task):
    def doit(self):
        print("taskA")


class SubTaskB(Task):
    def doit(self):
        print("taskB")

