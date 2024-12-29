# ...

# bibliotheken laden
import time
import datetime
import config as myC

# klasse definieren
class timestamp:

    def __init__(self):
        self.stamp = ''

    def start(self):
        if 'CLASS' in myC.theLoglevels: print(f"TIMESTAMP.PY > Timestamp gestartet")

    def generateTS(self):
        self.stamp = time.time()
        time.sleep(0.0001)
        return self.stamp

    def generateDT(self):
        self.stamp = datetime.datetime.now()
        return self.stamp

    def differenceTS(self, theTS1, theTS2):
        return (theTS2 - theTS1)

    def nowDate(self, ):
        return (datetime.date.today().strftime("%d.%m.%Y"))

    def nowTime(self, ):
        return (datetime.datetime.now().strftime("%H:%M:%S"))

    def nowDateForFile(self, ):
        return (datetime.date.today().strftime("%Y%m%d"))

    def nowTimeForFile(self, ):
        return (datetime.datetime.now().strftime("%H%M%S"))