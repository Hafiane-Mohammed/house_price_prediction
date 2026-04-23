import sys
import numpy as np
import pandas as pd
import joblib

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtWidgets import QGraphicsOpacityEffect

# 🔥 نفس function لي كانت ف training
def handle_outliers(X):
    import numpy as np

    X = np.array(X)
    X = X.copy()

    for i in range(X.shape[1]):
        col = X[:, i]

        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        col = np.clip(col, lower, upper)

        if (col > 0).all():
            col = np.log1p(col)

        X[:, i] = col

    return X

model = joblib.load("model.pkl")


class PriceApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart House Price Prediction")
        self.setGeometry(200, 200, 500, 650)
        self.setWindowIcon(QIcon("Image21.png"))

        # fade animation
        self.effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.effect)

        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(1000)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.start()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # IMAGE
        image = QLabel()
        pixmap = QPixmap("Image21.png")
        pixmap = pixmap.scaled(120, 120)
        image.setPixmap(pixmap)
        image.setAlignment(Qt.AlignCenter)
        layout.addWidget(image)

        # TITLE
        title = QLabel("🏠 Smart House Price Prediction")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:18px; font-weight:bold;")
        layout.addWidget(title)

        # INPUTS
        layout.addWidget(QLabel("Surface maison (m²)"))
        self.area = QLineEdit()
        self.area.setPlaceholderText("Ex: 120")
        layout.addWidget(self.area)

        layout.addWidget(QLabel("Chambres"))
        self.bedrooms = QComboBox()
        self.bedrooms.addItems([str(i) for i in range(1, 11)])
        layout.addWidget(self.bedrooms)

        layout.addWidget(QLabel("Salles de bain"))
        self.bathrooms = QComboBox()
        self.bathrooms.addItems([
            "0.5 = WC seulement",
            "1 = salle de bain",
            "1.5 = salle de bain + WC",
            "2 = 2 salles de bain",
            "2.5 = 2 salles + WC",
            "3 = 3 salles"
        ])
        layout.addWidget(self.bathrooms)

        layout.addWidget(QLabel("Qualité (Grade 1 - 13)"))
        self.grade = QSlider(Qt.Horizontal)
        self.grade.setMinimum(1)
        self.grade.setMaximum(13)
        self.grade.setValue(7)
        layout.addWidget(self.grade)

        self.grade_label = QLabel("Grade: 7")
        self.grade.valueChanged.connect(self.update_grade_label)
        layout.addWidget(self.grade_label)

        layout.addWidget(QLabel("Surface étage supérieur (m²)"))
        self.above = QLineEdit()
        layout.addWidget(self.above)

        layout.addWidget(QLabel("Surface voisins (m²)"))
        self.nearby = QLineEdit()
        layout.addWidget(self.nearby)

        layout.addWidget(QLabel("Étages"))
        self.floors = QComboBox()
        self.floors.addItems(["1", "2", "3", "4"])
        layout.addWidget(self.floors)

        # BUTTON
        btn = QPushButton("🔮 Predict Price")
        btn.clicked.connect(self.predict_price)
        layout.addWidget(btn)

        # RESULT
        self.result = QLabel("Predicted Price: ---")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("font-size:16px; font-weight:bold;")
        layout.addWidget(self.result)

        self.setLayout(layout)

    def update_grade_label(self):
        self.grade_label.setText(f"Grade: {self.grade.value()}")


    def predict_price(self):
        try:
            if self.area.text() == "" or self.above.text() == "" or self.nearby.text() == "":
                self.result.setText("⚠️ Please fill all fields")
                return

            # 🔥 مهم: DataFrame بنفس features ديال model
            features = pd.DataFrame([{
                "sqft_living": float(self.area.text()),
                "bedrooms": float(self.bedrooms.currentText()),
                "bathrooms": float(self.bathrooms.currentText().split(" = ")[0]),
                "grade": float(self.grade.value()),
                "sqft_above": float(self.above.text()),
                "sqft_living15": float(self.nearby.text()),
                "floors": float(self.floors.currentText())
            }])

            prediction = model.predict(features)[0]

            if prediction < 300000:
                level = "Low 💰"
            elif prediction < 800000:
                level = "Medium 💰💰"
            else:
                level = "High 💰💰💰"

            self.result.setText(f"Price: ${round(prediction, 2)} ({level})")

        except Exception as e:
            print(e)
            self.result.setText("⚠️ Invalid input!")


# RUN
app = QApplication(sys.argv)
window = PriceApp()
window.show()
sys.exit(app.exec_())