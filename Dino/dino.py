import os
import time
import random

import numpy as np
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# =====================================================================
#  НАСТРОЙКИ
# =====================================================================
TRAIN_MODE = True    # True — обучаем, False — просто играем по готовой таблице
LOAD_MODEL = True    # загружать сохранённую таблицу, если она есть
SAVE_MODEL = True    # сохранять таблицу после каждой игры

MODEL_FILE = "q_table_v3.npy"

EPISODES = 1000
MAX_STEPS = 10000     # предел шагов в одной игре

# --- Q-learning ---
ALPHA = 0.1          # скорость обучения
GAMMA = 0.9          # важность будущих наград
# Доля случайных действий. Здесь 0: агент и так пробует все действия благодаря
# оптимистичной начальной таблице (см. ниже), а случайные нажатия только
# убивают динозавра. Если обучение застрянет, поставьте EPSILON_START = 0.02.
EPSILON_START = 0.0
EPSILON_MIN = 0.0
EPSILON_DECAY = 0.995  # epsilon умножается на это число после каждой игры

# --- Награды ---
ALIVE_REWARD = 1.0       # за каждый шаг, который динозавр жив
DEATH_PENALTY = -100.0   # за смерть
ACTION_COST = 0.05       # маленький штраф за прыжок/приседание: не дёргаться зря

# --- Ускорение обучения ---
# 1) Оптимистичная начальная таблица: все действия сначала кажутся хорошими,
#    поэтому агент по очереди пробует каждое из них.
OPTIMISTIC_INIT = ALIVE_REWARD / (1.0 - GAMMA)
# 2) После смерти заново проигрываем последние шаги в обратном порядке,
#    чтобы штраф быстро дошёл до ошибочного действия (а не за десятки игр).
REPLAY_LEN = 40
REPLAY_PASSES = 6
# 3) В начале каждой игры случайная скорость: птицы (и приседание)
#    появляются сразу, а не после 40 секунд игры.
RANDOM_START_SPEED = True
START_SPEED_MIN = 6.0
START_SPEED_MAX = 11.0

# --- Время ---
STEP_DELAY = 0.04    # секунд на один шаг агента (постоянный ритм)

# =====================================================================
#  ДЕЙСТВИЯ И СОСТОЯНИЯ
# =====================================================================
ACTION_NOTHING = 0
ACTION_JUMP = 1
ACTION_DUCK = 2
NUM_ACTIONS = 3
ACTION_NAMES = ["НИЧЕГО", "ПРЫГАТЬ", "ПРИСЕСТЬ"]

# Состояние = (в воздухе?, время до препятствия, тип препятствия)
#
# Время до препятствия = расстояние / скорость игры (в кадрах).
# Корзины (кадры до столкновения), от далёкого к близкому:
#   0: >45   1: 32-45   2: 26-32   3: 21-26   4: 16-21   5: 11-16   6: <11
TTC_EDGES = [45, 32, 26, 21, 16, 11]
NUM_DIST = len(TTC_EDGES) + 1  # 7

# Тип: 0 = кактус, 1 = низкая птица (прыгать),
#      2 = высокая/средняя птица (приседать)
NUM_KINDS = 3
NUM_AIR = 2  # 0 = на земле, 1 = в воздухе
NUM_STATES = NUM_AIR * NUM_DIST * NUM_KINDS  # 42

KIND_NAMES = ["Кактус", "Птица низко", "Птица высоко"]
AIR_NAMES = ["земля", "воздух"]

# Приседание в игре работает, пока зажата клавиша «вниз»
is_ducking = False

# =====================================================================
#  JavaScript, который выполняется внутри страницы с игрой
# =====================================================================
JS_GET_STATE = """
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return { status: "NO_RUNNER" };
    if (runner.crashed) return { status: "DEAD" };
    if (!runner.playing) return { status: "NOT_PLAYING" };

    const dino = runner.tRex;
    const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
    const speed = runner.currentSpeed || 6;
    const jumping = !!dino.jumping;

    const widthOf = o => o.width || (o.typeConfig && o.typeConfig.width) || 20;

    // Ближайшее препятствие, которое ещё не пройдено
    const obstacle = obstacles.find(o => o.xPos + widthOf(o) > dino.xPos);

    if (!obstacle) {
        return { status: "OK", distance: 1000, kind: 0,
                 speed: speed, jumping: jumping };
    }

    let kind = 0;  // кактус
    if (obstacle.typeConfig && obstacle.typeConfig.type === "PTERODACTYL") {
        // yPos: 100 — низко, 75 и 50 — выше
        kind = obstacle.yPos > 90 ? 1 : 2;
    }

    return { status: "OK", distance: obstacle.xPos - dino.xPos,
             kind: kind, speed: speed, jumping: jumping };
}
"""

# Счёт, как на экране игры
JS_GET_SCORE = """
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return 0;
    return Math.round((runner.distanceRan || 0) * 0.025);
}
"""

JS_SET_SPEED = """
(value) => {
    const runner = window.Runner && window.Runner.instance_;
    if (runner && runner.playing && !runner.crashed) {
        runner.currentSpeed = value;
    }
}
"""

# =====================================================================
#  Q-ТАБЛИЦА
# =====================================================================


def new_q_table():
    return {s: [OPTIMISTIC_INIT] * NUM_ACTIONS for s in range(NUM_STATES)}


q_table = new_q_table()


def load_q_table():
    global q_table
    if LOAD_MODEL and os.path.exists(MODEL_FILE):
        loaded = np.load(MODEL_FILE, allow_pickle=True).item()
        valid = (
            len(loaded) == NUM_STATES
            and all(len(v) == NUM_ACTIONS for v in loaded.values())
        )
        if valid:
            q_table = loaded
            print("Загружена сохранённая Q-таблица.")
            return
        print("Сохранённая Q-таблица другого формата — создаём новую.")
    else:
        print("Q-таблица создана с нуля.")
    q_table = new_q_table()


def save_q_table():
    if not SAVE_MODEL:
        return
    # Сначала пишем во временный файл, чтобы Ctrl+C не испортил таблицу
    tmp_file = MODEL_FILE + ".tmp.npy"
    np.save(tmp_file, q_table)
    os.replace(tmp_file, MODEL_FILE)


# =====================================================================
#  СОСТОЯНИЕ
# =====================================================================


def encode_state(in_air, dist_bucket, kind):
    return (in_air * NUM_DIST + dist_bucket) * NUM_KINDS + kind


def decode_state(state):
    kind = state % NUM_KINDS
    rest = state // NUM_KINDS
    dist_bucket = rest % NUM_DIST
    in_air = rest // NUM_DIST
    return in_air, dist_bucket, kind


def time_bucket(ttc):
    """Номер корзины по времени до препятствия (в кадрах)."""
    for i, edge in enumerate(TTC_EDGES):
        if ttc > edge:
            return i
    return NUM_DIST - 1


def observe(page):
    """Возвращает (состояние, статус).

    Статус "OK" — игра идёт, состояние от 0 до 41.
    Статус "DEAD" — динозавр разбился. Другие статусы — игра не запущена.
    В обоих последних случаях состояние равно -1.
    """
    try:
        result = page.evaluate(JS_GET_STATE)
    except Exception as error:
        return -1, f"ERROR: {error}"

    status = result["status"]
    if status != "OK":
        return -1, status

    speed = max(result["speed"], 1.0)
    ttc = result["distance"] / speed
    in_air = 1 if result["jumping"] else 0

    state = encode_state(in_air, time_bucket(ttc), result["kind"])
    return state, "OK"


def get_score(page):
    try:
        return int(page.evaluate(JS_GET_SCORE))
    except Exception:
        return 0


# =====================================================================
#  ВЫБОР ДЕЙСТВИЯ И ОБНОВЛЕНИЕ ТАБЛИЦЫ
# =====================================================================


def choose_action(state, epsilon):
    # Иногда пробуем случайное действие (исследование)
    if TRAIN_MODE and random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)

    values = q_table[state]
    best = max(values)
    best_actions = [a for a, v in enumerate(values) if v >= best - 1e-9]

    # При обучении выбираем случайно среди равных, при игре — первое
    if TRAIN_MODE:
        return random.choice(best_actions)
    return best_actions[0]


def update_q_table(state, action, reward, next_state, terminal):
    current_q = q_table[state][action]
    max_next_q = 0.0 if terminal else max(q_table[next_state])

    q_table[state][action] = current_q + ALPHA * (
        reward + GAMMA * max_next_q - current_q
    )


def replay_backwards(trajectory):
    """Проигрываем конец игры задом наперёд.

    Штраф за смерть сначала попадает в последний шаг, затем в предыдущий
    (уже с учётом нового значения) и так далее. За один проход он доходит
    до действия, которое на самом деле было ошибкой.
    """
    tail = trajectory[-REPLAY_LEN:]
    for _ in range(REPLAY_PASSES):
        for state, action, reward, next_state, terminal in reversed(tail):
            update_q_table(state, action, reward, next_state, terminal)


# =====================================================================
#  УПРАВЛЕНИЕ ДИНОЗАВРОМ
# =====================================================================


def release_duck(page):
    """Отпускаем клавишу «вниз», если динозавр приседает."""
    global is_ducking
    if is_ducking:
        try:
            page.keyboard.up("ArrowDown")
        except Exception:
            pass
        is_ducking = False


def apply_action(page, action):
    global is_ducking

    if action == ACTION_DUCK:
        if not is_ducking:
            page.keyboard.down("ArrowDown")
            is_ducking = True
        return

    # Любое другое действие — сначала встаём
    release_duck(page)

    if action == ACTION_JUMP:
        page.keyboard.press("Space")


def start_game(page, settle=0.0):
    """Запускает новую игру. Возвращает (состояние, статус)."""
    release_duck(page)

    page.locator("canvas").click()
    time.sleep(0.2)
    page.keyboard.press("Space")
    time.sleep(settle)

    # Ждём, пока игра реально начнётся (не больше 3 секунд)
    state, status = -1, "NOT_PLAYING"
    for _ in range(30):
        state, status = observe(page)
        if state != -1:
            break
        if status == "DEAD":
            # Игра ещё в состоянии «разбился» — перезапускаем ещё раз
            page.locator("canvas").click()
            time.sleep(0.2)
            page.keyboard.press("Space")
        time.sleep(0.1)

    if state == -1:
        return state, status

    # Случайная стартовая скорость, чтобы птицы встречались чаще
    if TRAIN_MODE and RANDOM_START_SPEED:
        speed = random.uniform(START_SPEED_MIN, START_SPEED_MAX)
        page.evaluate(JS_SET_SPEED, speed)
        state, status = observe(page)

    return state, status


# =====================================================================
#  ОДНА ИГРА
# =====================================================================


def run_episode(page, epsilon, settle=0.0):
    """Играет одну игру. Возвращает словарь с итогами или None."""
    state, status = -1, "NOT_PLAYING"
    for _ in range(3):  # пара попыток на случай, если игра успела разбиться
        state, status = start_game(page, settle)
        if state != -1:
            break
    if state == -1:
        print(f"Игра не запустилась. Причина: {status}")
        return None

    trajectory = []
    total_reward = 0.0
    died = False
    steps = 0
    next_tick = time.perf_counter()

    for steps in range(1, MAX_STEPS + 1):
        action = choose_action(state, epsilon)
        apply_action(page, action)

        # Шаг всегда одной длины, как бы ни менялась задержка Python
        next_tick += STEP_DELAY
        wait = next_tick - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            next_tick = time.perf_counter()

        next_state, status = observe(page)

        if status == "OK":
            reward = ALIVE_REWARD
            if action != ACTION_NOTHING:
                reward -= ACTION_COST
            terminal = False
        elif status == "DEAD":
            reward = DEATH_PENALTY
            terminal = True
        else:
            # Что-то пошло не так (не смерть): шаг не учитываем
            print(f"Игра остановилась: {status}")
            steps -= 1
            break

        if TRAIN_MODE:
            update_q_table(state, action, reward, next_state, terminal)
            trajectory.append((state, action, reward, next_state, terminal))

        total_reward += reward

        if terminal:
            died = True
            break

        state = next_state

    release_duck(page)

    if TRAIN_MODE and died:
        replay_backwards(trajectory)

    return {
        "steps": steps,
        "reward": total_reward,
        "score": get_score(page),
        "died": died,
    }


# =====================================================================
#  ВЫВОД
# =====================================================================


def print_q_table():
    print("\nИтоговая Q-таблица (лучшее действие в каждом состоянии):")
    print("Корзина времени: 0=далеко ... 6=вплотную")

    for state, values in q_table.items():
        in_air, dist_bucket, kind = decode_state(state)
        label = (
            f"[{AIR_NAMES[in_air]:6}] "
            f"время={dist_bucket} {KIND_NAMES[kind]:13}"
        )

        if max(values) - min(values) < 1e-6:
            print(f"{label}: не изучено")
            continue

        best_action = int(np.argmax(values))
        values_str = ", ".join(f"{v:6.1f}" for v in values)
        print(f"{label}: [{values_str}] -> {ACTION_NAMES[best_action]}")


# =====================================================================
#  ГЛАВНАЯ ПРОГРАММА
# =====================================================================


def main():
    global is_ducking

    print("Запускаем программу...")
    load_q_table()

    best_score = 0
    recent_scores = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--mute-audio",
            ]
        )

        page = browser.new_page(
            viewport={
                "width": 800,
                "height": 600
            }
        )

        print("Открываем игру...")
        page.goto(GAME_URL, wait_until="networkidle")

        print("Ждём загрузку игры...")
        time.sleep(2.5)

        first_game = True  # после загрузки страницы даём игре время на интро

        try:
            for episode in range(1, EPISODES + 1):
                if TRAIN_MODE:
                    epsilon = max(
                        EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (episode - 1)
                    )
                else:
                    epsilon = 0.0

                print(f"\n--- Игра №{episode} ---")
                result = run_episode(
                    page, epsilon, settle=1.0 if first_game else 0.0
                )

                if result is None:
                    print("Перезагружаем страницу...")
                    page.reload(wait_until="networkidle")
                    is_ducking = False
                    time.sleep(2.5)
                    first_game = True
                    continue

                first_game = False

                best_score = max(best_score, result["score"])
                recent_scores.append(result["score"])
                recent_scores = recent_scores[-10:]
                average = sum(recent_scores) / len(recent_scores)

                end_note = "" if result["died"] else " (лимит шагов)"
                print(
                    f"Очки: {result['score']}{end_note}. "
                    f"Шагов: {result['steps']}. "
                    f"Награда: {result['reward']:.1f}. "
                    f"Среднее за 10 игр: {average:.0f}. "
                    f"Рекорд: {best_score}. "
                    f"epsilon={epsilon:.3f}"
                )

                if TRAIN_MODE:
                    save_q_table()

        except KeyboardInterrupt:
            print("\nОстановлено пользователем (Ctrl+C).")

        try:
            release_duck(page)
        except Exception:
            pass

        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!" if TRAIN_MODE else "\nИГРА ЗАВЕРШЕНА!")

        print_q_table()

        if TRAIN_MODE:
            save_q_table()
            print(f"\nQ-таблица сохранена в файл {MODEL_FILE}")

        input("\nНажми Enter, чтобы закрыть браузер...")

        browser.close()


if __name__ == "__main__":
    main()