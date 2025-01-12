from subprocess import STARTUPINFO

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#########################################################
# DATEN laden
#########################################################

file_path = 'sihl_2020_2024_daily_original.csv'
df = pd.read_csv(file_path,delimiter=';')

#########################################################
# DATEN vorbereiten
#########################################################

df.rename(columns={'Temperatur': 'TE'}, inplace=True)
df.rename(columns={'Abflussmenge': 'AM'}, inplace=True)
df.rename(columns={'Wasserstand': 'WS'}, inplace=True)

df['TE'] = df['TE'].round(1)
df['AM'] = df['AM'].round(1)
df['WS'] = df['WS'].round(1)
print(df.head())
print(df.describe())

myTEbefore=df.loc[0, 'TE']
myAMbefore=df.loc[0, 'AM']
myWSbefore=df.loc[0, 'WS']

for mySchritt, row in df.iterrows():
    df.at[mySchritt, 'Datum'] = int(mySchritt)
    df.at[mySchritt, 'dTE'] = row['TE'] - myTEbefore
    df.at[mySchritt, 'dWS'] = row['WS'] - myWSbefore
    df.at[mySchritt, 'dAM'] = row['AM'] - myAMbefore
    myTEbefore=row['TE']
    myWSbefore=row['WS']
    myAMbefore=row['AM']

df['dTE'] = df['dTE'].round(1)
df['dWS'] = df['dWS'].round(1)
df['dAM'] = df['dAM'].round(1)

print(df.head())
print(df.describe())

#########################################################
# MÖGLICHE AKTION-POLICIES
#########################################################

def actionRandomPosition(myDF,mySchritt):
    return random.choice(myPositions)

def actionRandomMove(myDF,mySchritt):
    return random.choice(myMoves)

def actionMove(myDF,mySchritt):
    myActionTE = 0 # einzelner ActionValue aus Analyse Temperatur
    myActionAM = 0 # einzelner ActionValue aus Analyse Abflussmenge
    myActionWS = 0 # einzelner ActionValue aus Analyse Wasserstand
    myActionValue = 0 # Summe aller ActionsValues
    myActionDecision = 0 # Definitive Action welche sich aus ActionValues ergeben

    # ACTION for TE
    if myDF.loc[mySchritt,'TE'] < 15:
        myActionTE=-1
    elif myDF.loc[mySchritt,'TE'] < 18:
        myActionTE=1
    elif myDF.loc[mySchritt,'TE'] < 50:
        myActionTE=-1
    if -0.1 <myDF.loc[mySchritt,'dTE'] < 0.1:
        myActionTE=0
    myDF.at[mySchritt, 'aTE'] = myActionTE

    # ACTION for AM
    if myDF.loc[mySchritt,'AM'] < 2:
        myActionAM=-1
    elif myDF.loc[mySchritt,'AM'] < 4:
        myActionAM=1
    elif myDF.loc[mySchritt,'AM'] < 1000:
        myActionAM=-1
    myDF.at[mySchritt, 'aAM'] = myActionAM

    # ACTION for WS
    if myDF.loc[mySchritt,'WS'] < 697.5:
        myActionWS=-1000
    else:
        myActionWS=1
    myDF.at[mySchritt, 'aWS'] = myActionWS
    # ACTION for SE

    # ACTION für alle
    myActionValue = (2*myActionTE + 3*myActionAM + 1*myActionWS)/5
    if myActionValue < -0.5:
        myActionDecision = -1
    elif myActionValue < 0.4:
        myActionDecision = 0
    else:
        myActionDecision = 1
    myDF.at[mySchritt, 'AV'] = myActionValue
    myDF.at[mySchritt, 'AD'] = myActionDecision

def actionPosition(myDF,mySchritt):
    print(myDF.loc[mySchritt, 'TE'])
    return random.choice(myMoves)

#########################################################
# SCHWIMMEN = EINE EPISODE
#########################################################

def swimm(myDF):
    for theSchritt, theRow in myDF.iterrows():
        if myPolicy=='actionRandomMove':
            myMove=actionRandomMove(myDF,theSchritt)
        elif myPolicy == 'actionRandomPosition':
            myMove = actionRandomPosition(myDF, theSchritt)
        elif myPolicy == 'actionMove':
            myMove = actionMove(myDF, theSchritt)
        elif myPolicy == 'actionPosition':
            myMove = actionPosition(myDF, theSchritt)
        else:
            myMove=actionRandomPosition(myDF,theSchritt)
 #       showEnvSchritt(myDF, theSchritt)

def showEnvSchritt(myDF,theSchritt):
    print(f"SCHRITT {theSchritt} > TE={myDF.loc[mySchritt,'TE']} dTE={myDF.loc[mySchritt,'dTE'].round(1)} aTE={myDF.loc[mySchritt,'aTE']}")

def showEnvAll(myDF):
    for theSchritt, theRow in myDF.iterrows():
        print(f"SCHRITT {theSchritt} > TE={theRow['TE']} dTE={theRow['dTE']} aTE={theRow['aTE']} + WS={theRow['WS']} dWS={theRow['dWS']} aWS={theRow['aWS']} + AM={theRow['AM']} dAM={theRow['dAM']} aAM={theRow['aAM']} >>> AV={theRow['AV']} AD={theRow['AD']}")

##############################################
# PROG START
##############################################

myPositions=[0,1,2,3,4,5,6,7,8,9] # Agent kann zehn Positionen im Fluss einnehmen. 0=Ufer/langsam 9=Mitte/schnell
myMoves=[-1,0,1] # mögliche Aktionen vom Typ Move (-1= richtung Ufer bewegen, 0=stay, 1=richtung Mitte gehen
myPolicy='actionMove' # aktuelle Policy
swimm(df) # eine Episode durchspielen
showEnvAll(df) # zeigen Environments inkl. Aktionen

exit()

# Plot Temperatur
plt.plot(df['Datum'], df['TE'],  label="TE" , color='blue')
plt.plot(df['Datum'], df['dTE'], label="dTE", color='red')
plt.xlabel("Schritte")
plt.ylabel("Temperatur")
plt.title("Entwicklung Temperatur")
plt.legend()
plt.grid()
plt.savefig("plot_TE.png")
plt.show()

# Plot Wasserstand
plt.plot(df['Datum'], df['WS']-690,  label="WS" , color='blue')
plt.plot(df['Datum'], df['dWS'], label="dWS", color='red')
plt.xlabel("Schritte")
plt.ylabel("Wasserstand")
plt.title("Entwicklung Wasserstand")
plt.legend()
plt.grid()
plt.savefig("plot_WS.png")
plt.show()

# Plot Abflussmenge
plt.plot(df['Datum'], df['AM']+10,  label="AM" , color='blue')
plt.plot(df['Datum'], df['dAM'], label="dAM", color='red')
plt.xlabel("Schritte")
plt.ylabel("Abflussmenge")
plt.title("Entwicklung Abflussmenge")
plt.legend()
plt.grid()
plt.savefig("plot_AM.png")
plt.show()

