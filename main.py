import time
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, Button, GameVariable
from RewardHandler import SpeedrunRewardHandler

WAD_PATH = "DOOM.WAD"
MAP = "E1M1"

def make_game(wad_path: str, map_name: str) -> DoomGame:
    game = DoomGame()
    game.set_doom_scenario_path(wad_path)
    game.set_doom_map(map_name)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    # Define a small action space (forward, turn left, turn right, attack)
    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.set_episode_timeout(2000)
    game.init()
    return game

def run_episode(game: DoomGame, render_delay: float = 0.09):
    game.new_episode()
    rewardHandler = SpeedrunRewardHandler(game)
    totalReward = 0
    while not game.is_episode_finished():
        # very simple policy: move forward
        action = [1, 0, 0, 0]
        game.make_action(action, 4)  # repeat for a few ticks for smoother motion
        totalReward += rewardHandler.calculate_reward()
        #print(totalReward)
        time.sleep(render_delay)
    print("Episode finished. Total reward:", game.get_total_reward())

if __name__ == "__main__":
    g = make_game(WAD_PATH, MAP)
    try:
        run_episode(g)
    finally:
        g.close()