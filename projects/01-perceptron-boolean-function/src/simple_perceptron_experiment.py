from itertools import product, combinations
import matplotlib.pyplot as plt


class PerceptronBooleanExperiment:
    # Простой класс: вся логика лабы в одном месте
    def __init__(self, eta=0.3, max_epochs=1000):
        self.eta = eta
        self.max_epochs = max_epochs

        # Все 16 входных наборов
        self.X = list(product([0, 1], repeat=4))

        # Целевой вектор считаем по формуле
        self.T = [self.F(*x) for x in self.X]

    # Булева функция варианта 25
    @staticmethod
    def F(x1, x2, x3, x4):
        return int(not ((x1 and x2) or (not x3) or (not x4)))

    # mode='thr' -> пороговая ФА
    # mode='smooth' -> 0.5*(net/(1+|net|)+1)
    def y_and_d(self, net, mode):
        if mode == 'thr':
            y = 1 if net >= 0 else 0
            d = 1.0
        else:
            out = 0.5 * (net / (1 + abs(net)) + 1)
            y = 1 if out >= 0.5 else 0
            d = 1 / (2 * (1 + abs(net)) ** 2)
        return y, d

    # Проверка сети на всех 16 наборах
    def evaluate(self, w, mode):
        Y = []
        for x in self.X:
            net = w[0] + w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[4] * x[3]
            y, _ = self.y_and_d(net, mode)
            Y.append(y)

        E = sum(int(y != t) for y, t in zip(Y, self.T))
        return Y, E

    # Обучение на выбранных индексах train_ids
    def train(self, train_ids, mode):
        w = [0.0, 0.0, 0.0, 0.0, 0.0]
        history = []  # (epoch, w_copy, Y_string, E)

        # Строка k=0 (до обучения)
        Y0, E0 = self.evaluate(w, mode)
        history.append((0, w[:], ''.join(map(str, Y0)), E0))

        for epoch in range(1, self.max_epochs + 1):
            for i in train_ids:
                x = self.X[i]
                t = self.T[i]
                inp = [1, x[0], x[1], x[2], x[3]]

                net = sum(w[j] * inp[j] for j in range(5))
                y, d = self.y_and_d(net, mode)
                delta = t - y

                for j in range(5):
                    if mode == 'thr':
                        w[j] += self.eta * delta * inp[j]
                    else:
                        w[j] += self.eta * delta * d * inp[j]

            Y, E = self.evaluate(w, mode)
            history.append((epoch, w[:], ''.join(map(str, Y)), E))

            if E == 0:
                break

        return w, history

    # Поиск минимального подмножества
    def find_min_subset(self, mode):
        for size in range(1, 17):
            for ids in combinations(range(16), size):
                w, h = self.train(ids, mode)
                if h[-1][3] == 0:
                    return ids, w, h
        return None, None, None

    def show_plot(self, history, title):
        epochs = [row[0] for row in history]
        errors = [row[3] for row in history]

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, errors, marker='o')
        plt.xlabel('Номер эпохи')
        plt.ylabel('Суммарная ошибка E(k)')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    def print_truth_table(self):
        print('\nТаблица истинности:')
        for i, x in enumerate(self.X):
            print(i, x, 'F =', self.T[i])

    def print_short_result(self, name, w, h):
        print(f'\n{name}')
        print('Эпох:', len(h) - 1)  # без k=0
        print('Весы:', [round(v, 6) for v in w])
        print('Y:', h[-1][2])
        print('E:', h[-1][3])

    # Полная таблица шагов обучения, чтобы видеть каждую эпоху
    def print_full_history(self, title, history):
        print(f'\n{title}')
        print('k | w0 | w1 | w2 | w3 | w4 | Y(k) | E(k)')
        for k, w, y, e in history:
            print(
                f'{k:>2} | '
                f'{w[0]:>8.6f} | {w[1]:>8.6f} | {w[2]:>8.6f} | {w[3]:>8.6f} | {w[4]:>8.6f} | '
                f'{y} | {e}'
            )

    def run(self):
        print('Вариант: 25')
        print('F(x1,x2,x3,x4)=NOT((x1 AND x2) OR (NOT x3) OR (NOT x4))')
        print('T =', ''.join(map(str, self.T)))

        self.print_truth_table()

        # Полная выборка
        w_thr_full, h_thr_full = self.train(range(16), 'thr')
        w_smo_full, h_smo_full = self.train(range(16), 'smooth')

        self.print_short_result('ФА 1 (пороговая), полная выборка', w_thr_full, h_thr_full)
        self.print_short_result('ФА 2 (гладкая), полная выборка', w_smo_full, h_smo_full)

        # Печатаем полные таблицы шагов
        self.print_full_history('Таблица эпох: ФА 1, полная выборка', h_thr_full)
        self.print_full_history('Таблица эпох: ФА 2, полная выборка', h_smo_full)

        # Минимальные подмножества
        ids_thr, w_thr_min, h_thr_min = self.find_min_subset('thr')
        ids_smo, w_smo_min, h_smo_min = self.find_min_subset('smooth')

        print('\nМинимальный набор для ФА 1:', ids_thr)
        self.print_short_result('ФА 1, минимальная выборка', w_thr_min, h_thr_min)

        print('\nМинимальный набор для ФА 2:', ids_smo)
        self.print_short_result('ФА 2, минимальная выборка', w_smo_min, h_smo_min)

        self.print_full_history('Таблица эпох: ФА 1, минимальная выборка', h_thr_min)
        self.print_full_history('Таблица эпох: ФА 2, минимальная выборка', h_smo_min)

        # Four learning-curve plots for the experiment variants.
        self.show_plot(h_thr_full, 'E(k), ФА 1, полная выборка')
        self.show_plot(h_smo_full, 'E(k), ФА 2, полная выборка')
        self.show_plot(h_thr_min, 'E(k), ФА 1, минимальная выборка')
        self.show_plot(h_smo_min, 'E(k), ФА 2, минимальная выборка')


if __name__ == '__main__':
    lab = PerceptronBooleanExperiment(eta=0.3, max_epochs=1000)
    lab.run()
