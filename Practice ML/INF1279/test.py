#Returning Multiple Values using Tuples
def multiple():
    operation = "Sum" 
    total = 5+10
    return operation, total;

def main():
    operation, total = multiple()
    print(operation)
    #Output = Sum 15

main()
