import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#дані
data = pd.DataFrame({
    'K': [1420, 1510, 1470, 1450, 1500, 1560, 1580, 1405, 1550, 1440],
    'L': [2160, 2195, 2020, 2130, 2200, 2220, 2150, 2190, 2235, 2180],
    'F': [5105, 5260, 5128, 5115, 5327, 5280, 5324, 5116, 5186, 5142]
})

# Логарифмічне перетворення
log_K = np.log(data['K'])
log_L = np.log(data['L'])
log_F = np.log(data['F'])

# Регресія для визначення параметрів
X = np.column_stack((log_K, log_L, np.ones(len(log_K))))
beta = np.linalg.lstsq(X, log_F, rcond=None)[0]

alpha, beta, log_A = beta[0], beta[1], beta[2]

# Параметри вартості факторів виробництва (припущення)
A = np.exp(log_A)
r = 10  # Ціна капіталу
w = 20  # Ціна праці
p = 40  # припущення ціни продукту

# Вихідні дані
K = data['K']
L = data['L']
F = A * K**alpha * L**beta  # Обчислення функції виробництва

# Розрахунок прибутку
# Довгостроковий період Ver 1.0
# Profit_long = p * A * K**alpha * L**(1-alpha) - (r * K + w * L)
# Довгостроковий період 2.0
Profit_long = p * F - (r * K + w * L)
# Короткостроковий період, припустимо, що K є фіксованим
# K_fixed = K.mean()  # середнє значення капіталу
K_fixed = 1420  # Приймемо значення капіталу з першого спостереження
Profit_short = p * A * K_fixed**alpha * L**(1-alpha) - (r * K_fixed + w * L)

# Фірма в умовах монополії-монопсонії
# припустимо, що умови монополії-монопсонії впливають на ціну продукту
p_monopoly = p * 1.2  # припущення, що фірма може збільшити ціну
#Profit_monopoly = p_monopoly * A * K**alpha * L**(1-alpha) - (r * K + w * L)
# Фірма в умовах монополії-монопсонії
#Profit_monopoly = p_monopoly * F - (r * K + w * L)
wK_monopoly = r * 1.5  # Припустимо, що ціна капіталу збільшується
wL_monopoly = w * 1.2   # Припустимо, що ціна праці збільшується

Profit_monopoly = p_monopoly * F - (wK_monopoly * K + wL_monopoly * L)
# Пункт 1: Вивід результатів
results_1 = pd.DataFrame({
    'LnK': log_K,
    'LnL': log_L,
    'LnF': log_F
})

# Ефект масштабу та еластичність заміщення
scale_effect = alpha + beta
substitution_elasticity = 1  # Для Cobb-Douglas функції еластичність заміщення = 1

results_2 = pd.DataFrame({
    'alpha': [alpha],
    'beta': [beta],
    'Scale effect': [scale_effect],
    'Substitution elasticity': [substitution_elasticity]
})

# Візуалізація результатів
plt.figure(figsize=(12, 14))

plt.subplot(3, 1, 1)
plt.axis('off')
plt.title("Пункт 1: Логарифмічні значення")
plt.table(cellText=results_1.round(3).values, colLabels=results_1.columns, loc='center', cellLoc='center')

plt.subplot(4, 1, 2)
plt.axis('off')
plt.title("Пункт 2: Ефект масштабу та еластичність заміщення")
plt.table(cellText=results_2.round(3).values, colLabels=results_2.columns, loc='center', cellLoc='center')


results_3 = pd.DataFrame({
    'Період': ['Довгостроковий', 'Короткостроковий'],
    'p': [p, p],
    'wK': [r, r],
    'wL': [w, w],
    'Profit': [Profit_long.sum(), Profit_short.sum()]
})




# Відображення результатів для монополії-монопсонії
results_4 = pd.DataFrame({
    'p': [p_monopoly],
    'wK': [wK_monopoly],
    'wL': [wL_monopoly],
    'Profit': [Profit_monopoly.sum()],
    'F': [F.sum()]
})
# Візуалізація таблиці для пункту 3
plt.subplot(5, 1, 3)
plt.axis('off')
plt.title("Пункт 3: Фірма в умовах досконалої конкуренції")
table_3 = plt.table(cellText=results_3.values, colLabels=results_3.columns, loc='center', cellLoc='center')
table_3.auto_set_font_size(False)
table_3.set_fontsize(10)
table_3.scale(1.2, 1.2)

# Візуалізація таблиці для пункту 4
plt.subplot(5, 1, 4)
plt.axis('off')
plt.title("Пункт 4: Фірма в умовах монополії-монопсонії")
table_4 = plt.table(cellText=results_4.values, colLabels=results_4.columns, loc='center', cellLoc='center')
table_4.auto_set_font_size(False)
table_4.set_fontsize(10)
table_4.scale(1.2, 1.2)

plt.tight_layout()
plt.show()

