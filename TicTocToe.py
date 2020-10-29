import itertools
from colorama import Fore, Style, init

init()


def game_board(game_map, player_input=0, row=0, column=0, just_display=False):
    try:
        if game_map[row][column] != 0:
            print("This position is occupied! Choose another!")
            return game_map, False
        print("   " + "  ".join([str(i) for i in range(len(game_map))]))
        if not just_display:
            game_map[row][column] = player_input

        for cislo, row in enumerate(game_map):
            colored_row = ""
            for item in row:
                if item == 0:
                    colored_row += "   "
                elif item == 1:
                    colored_row += Fore.GREEN + ' X ' + Style.RESET_ALL
                elif item == 2:
                    colored_row += Fore.MAGENTA + ' O ' + Style.RESET_ALL
            print(cislo, colored_row)

        return game_map, True

    except IndexError as e:
        print("Did you attempt to play a row or column outside the range of 0,1 or 2?", e)
        return game_map, False
    except Exception as e:
        print("Something went very wrong!", e)
        return game_map, False


def win():
    def all_same(el):
        if el.count(el[0]) == len(el) and el[0] != 0:
            return True
        else:
            return False

    # Horizontal
    for row in game:
        print(row)
        if all_same(row):
            print(f"Player {row[0]} is the winner horizontally!")
            return True

    # Vertical
    for col in range(len(game)):
        check = []

        for row in game:
            check.append(row[col])

        if all_same(check):
            print(f"Player {check[0]} is the winner vertically!")
            return True

    # Diagonal
    diags = []
    for ix in range(len(game)):
        diags.append(game[ix][ix])
    if all_same(diags):
        print(f"Player {diags[0]} is the winner diagonally (\\)!")
        return True

    diags = []
    for col, row in enumerate(reversed(range(len(game)))):
        diags.append(game[row][col])
    if all_same(diags):
        print(f"Player {diags[0]} is the winner diagonally  (/)!")
        return True
    return False


play = True
players = [1, 2]
while play:
    game_size = int(input("What size game of tic tac toc do you want to play? "))
    game = [[0 for i in range(game_size)] for i in range(game_size)]
    game_won = False
    game, _ = game_board(game, just_display=True)
    player_choice = itertools.cycle([1, 2])
    while not game_won:
        current_player = next(player_choice)
        print(f"Current player: {current_player}")
        played = False

        while not played:
            column_choice = int(input("What column do you want to play? (0, 1 , 2): "))
            row_choice = int(input("What row do you want to play? (0, 1 , 2): "))
            game, played = game_board(game, current_player, row_choice, column_choice)

        if win():
            game_won = True
            again = input("The game is over, would you like to play again? (y/n)")
            if again.lower() == "y":
                print("Restarting")
            elif again.lower() == "n":
                print("Byeee :)")
                play = False
            else:
                print("Not a valid answer, so see you later!")
                play = False
