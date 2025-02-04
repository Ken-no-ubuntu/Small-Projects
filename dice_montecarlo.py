import random as rd
def dice_rll(ntrials):
    nsuccess = 0
    for x in range(ntrials):
        rolls =[rd.randint(1,6) for x in range(10)]
        for i in range(8):
            if rolls[i:i+3] == [3,3,3]:
                nsuccess += 1
                break
    prob = nsuccess / ntrials
    return prob
ntrials = 1000000
prob = dice_rll(ntrials)
print(f"The probability is {prob}")
