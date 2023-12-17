# Python Program to Check if a Date is Valid and Print the Incremented Date if it is
try:
    date = int(input("Enter The Date (as ddmmyyyy): "))

    if(len(str(date))==8) or (len(str(date))==7):
        print("\nDate is valid.\n")
        date = date + 1000000
        dat = date
        year = dat%10000
        dat = int(dat/10000)
        month = dat%100
        dat = int(dat/100)
    
    print(f"and Incremented Date is: {dat}-{month}-{year}\n")

except:
    print("\nInvalid Date.\n")
