from PyQt5.QtWidgets import QApplication, QMainWindow

from New_window import Ui_MainWindow
from figure_plot import FigurePlot


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui: Ui_MainWindow = Ui_MainWindow()  
        self.ui.setupUi(self)

        self.figure_plot_window = FigurePlot(self)  # tạo đối tượng self. để không bị xóa tự động khi biên dịch

if __name__ == "__main__":
    app = QApplication([])
    main_win = MainWindow()
    main_win.show()
    app.exec_()


