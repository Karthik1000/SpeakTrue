import sys
from random import randint

def generateOTP():
    global generatedOTP
    
    generatedOTP = randint(1000,8999)
    #print(generatedOTP)

def verifyOTP(OTP):
    global generatedOTP
    return (OTP == generatedOTP)

def attackAndStealOTP():
    #x = 
    global OTP
    print(verifyOTP(OTP))   
    print(OTP) 
    #print(generatedOTP)

    #print(verifyOTP(print(generateOTP)))
    
    #return (generatedOTP)

generatedOTP = 0
generateOTP()
print(generatedOTP)
otp = attackAndStealOTP()
print(verifyOTP(otp))