from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from UI.ui import Ui_MainWindow
import sys
from time import sleep
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

class MyThead(QObject):
    signal = pyqtSignal()
    Value = pyqtSignal(int)
    def __init__(self) -> None:
        super().__init__()
        self.signal.connect(self.Update)
        self.Value.connect(self.UpdateValue)
        self.run()

    def run(self):
        while True:
            print('Loading')
            sleep(1)
            self.signal.emit()
            self.Value.emit(1)
        
    def UpdateValue(self):
        i = 0
        while True:
            print(i)
            sleep(1)
            self.Value.emit(1)
            i += 1

    def Update(self):
        print('Update')

class MyApp(Ui_MainWindow):
    def __init__(self) -> None:
        super().setupUi(MainWindow)
        self.StartThread()

    def StartThread(self):
        self.myThread = MyThead()

obj = MyApp()

if __name__ == '__main__':
    MainWindow.show()
    sys.exit(app.exec_())