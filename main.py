import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Вихідні дані
K = np.array([1420, 1510, 1470, 1450, 1500, 1560, 1580, 1405, 1550, 1440])
L = np.array([2160, 2195, 2020, 2130, 2200, 2220, 2150, 2190, 2235, 2180])
F = np.array([5105, 5260, 5128, 5115, 5327, 5280, 5324, 5116, 5186, 5142])

# Логарифмування даних
log_K = np.log(K)
log_L = np.log(L)
log_F = np.log(F)

# Формування матриці незалежних змінних
X = np.vstack([np.ones(len(K)), log_K, log_L]).T  # Додаємо стовпчик одиниць для вільного члена

# Розрахунок коефіцієнтів регресії за допомогою методу найменших квадратів
coefficients = np.linalg.inv(X.T @ X) @ X.T @ log_F

# Вилучення параметрів
log_A = coefficients[0]
alpha = coefficients[1]
beta = coefficients[2]

# Ефект масштабу
scale_effect = alpha + beta

# Еластичність заміщення для мультиплікативної виробничої функції
substitution_elasticity = 1  # Для Cobb-Douglas функції

# Параметри вартості факторів виробництва (припущення)
A = np.exp(log_A)
r = 10  # Ціна капіталу
w = 20  # Ціна праці
optimal_K = (alpha / r) / ((alpha / r) + (beta / w)) * A ** (1 / (1 - alpha - beta))
optimal_L = (beta / w) / ((alpha / r) + (beta / w)) * A ** (1 / (1 - alpha - beta))

# Побудова графіків
Q = np.linspace(1, 100, 100)  # Обсяг продукції
K_cost = 2 * Q  # Витрати на капітал
L_cost = 3 * Q  # Витрати на працю
Total_cost = K_cost + L_cost  # Загальні витрати
Price = 10 - 0.05 * Q  # Ціна продукції

# Візуалізація
plt.figure(figsize=(12, 16))

# Таблиця з результатами
result_data = {
    "log(A)": [log_A],
    "alpha": [alpha],
    "beta": [beta],
    "Scale effect": [scale_effect],
    "Substitution elasticity": [substitution_elasticity],
    "Optimal K": [optimal_K],
    "Optimal L": [optimal_L]
}
result_table = pd.DataFrame(result_data)

plt.subplot(3, 1, 1)
plt.axis('tight')
plt.axis('off')
the_table = plt.table(cellText=result_table.values, colLabels=result_table.columns, cellLoc='center', loc='upper center', colWidths=[0.15]*len(result_table.columns))
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 1.2)

plt.subplot(3, 2, 3)
plt.plot(Q, K_cost, label='Капітал')
plt.plot(Q, L_cost, label='Праця')
plt.plot(Q, Total_cost, label='Загальні витрати')
plt.xlabel('Обсяг продукції')
plt.ylabel('Витрати')
plt.title('Витрати виробництва')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(Q, Price)
plt.xlabel('Обсяг продукції')
plt.ylabel('Ціна продукції')
plt.title('Цінова функція продукції')

plt.tight_layout()
plt.show()
