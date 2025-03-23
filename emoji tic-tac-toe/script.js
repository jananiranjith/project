class Player {
    /**
     * @constructor
     * @param {number} id Unique number of the player
     * @param {string} name The name of the player
     * @param {HTMLElement} emoji HTML I Element representing the emoji
     */
    constructor(id, name, emoji) {
      this.id = id;
      this.name = name;
      this.emoji = emoji;
    }
  }
  
  /**
   * @typedef GameState
   * @type {object}
   * @property {Player} turn Player who has the turn
   * @property {Array.<?Player>} marks Marks of the sqaures
   * @property {boolean} gameOver Game is over
   * @property {?Player} winner Player who is the winner
   */
  
  /**
   * Representing a game of Tic-Tac-Toe
   */
  class TicTacToe {
    /**
     * @constructor
     * @param {Player} player1 Player 1
     * @param {Player} player2 Player 2
     * @param {GameState} [state] Current state of the game
     */
    constructor(
      player1,
      player2,
      state = {
        turn: player1,
        marks: [...Array(9).fill(null)],
        gameOver: false,
        winner: null
      }
    ) {
      this.player1 = player1;
      this.player2 = player2;
      this.state = state;
    }
  
    /**
     * Depending on the current state of the game, it returns an object with information about the winning player and if the game has ended
     *
     *  @return {GameState} State Winner and Game Over
     */
    checkGameOver() {
      const state = this.state;
      const isWinner = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
      ].some(line => {
        return line.every(
          i => state.marks[i] && state.marks[i].id === state.turn.id
        );
      });
  
      const isOver = isWinner || state.marks.every(mark => mark);
  
      return {
        winner: isWinner ? state.turn : null,
        gameOver: isOver
      };
    }
  
    /**
     * Reverse the players turn
     *
     * @return {Player} Player with the turn
     */
    nextTurn() {
      const [player1, player2] = [this.player1, this.player2];
      try {
        switch (this.state.turn.id) {
          case player1.id:
            return { turn: player2 };
          case player2.id:
            return { turn: player1 };
          default:
            throw new Error("Turn's ID does not match the ID of any player");
        }
      } catch (error) {
        console.error(error);
      }
    }
  
    isAvaliableSquare(squarePosition) {
      return !this.state.marks[squarePosition];
    }
  
    /**
     * Play the specified square and carry the following actions:
     * 1. Marks the selected square
     * 2. Pass the turn
     * 3. Check if the game is finished
     *
     * This can only be executed if one of these two conditions is not met:
     * 1. The game is over
     * 2. The square is not avaliable
     *
     * @param {number} squarePosition Position (from 0 to 8) of the square where the player with the turn will play
     */
    play(squarePosition) {
      try {
        if (this.state.gameOver) {
          throw new Error("The game is over");
        }
        if (!this.isAvaliableSquare(squarePosition)) {
          throw new Error("Not avaliable square");
        }
  
        this.state.marks[squarePosition] = this.state.turn;
        this.state = {
          marks: this.state.marks,
          ...this.nextTurn(),
          ...this.checkGameOver()
        };
      } catch (error) {
        console.error(error);
      }
    }
  }
  
  class UI {
    /**
     * @constructor
     * @param {HTMLDivElement} htmlContainer
     */
    constructor(htmlContainer) {
      this.htmlContainer = htmlContainer;
    }
  
    static HIDE_CLASS_NAME = "hide";
  
    show() {
      this.htmlContainer.classList.remove(UI.HIDE_CLASS_NAME);
    }
  
    hide() {
      this.htmlContainer.classList.add(UI.HIDE_CLASS_NAME);
    }
  }
  
  class OptionsUI extends UI {
    /**
     * @constructor
     * @param {HTMLDivElement} htmlContainer Container
     * @param {NodeList} htmlsPlayerName Node list of players name html input elements
     * @param {NodeList} htmlsPlayerEmoji Node list of players emoji html span elements
     * @param {HTMLUListElement} htmlEmojis Emojis list from CSS Emoji library
     * @param {HTMLButtonElement} htmlOKButton Play button
     */
    constructor(
      htmlContainer,
      htmlsPlayerName,
      htmlsPlayerEmoji,
      htmlEmojis,
      htmlOKButton
    ) {
      super(htmlContainer);
      // Define properties
      this.htmlsPlayerName = htmlsPlayerName;
      this.htmlsPlayerEmoji = htmlsPlayerEmoji;
      this.htmlEmojis = htmlEmojis;
      this.htmlOKButton = htmlOKButton;
      // Execute necessary functions
      this.checkValidInput();
      this.initEvents();
    }
  
    static SELECTING_CLASS_NAME = "selecting";
    static EMOJI_TAG_NAME = "I";
    static EMOJI_CLASS_NAME = "em";
    static EMOJI_SVG_CLASS_NAME = "em-svg";
  
    static isPlayerSelecting(htmlElement) {
      return htmlElement.dataset.playerEmoji === OptionsUI.SELECTING_CLASS_NAME;
    }
  
    playerSelecting() {
      return [...this.htmlsPlayerEmoji].find(OptionsUI.isPlayerSelecting);
    }
  
    switchPlayers() {
      this.htmlsPlayerEmoji.forEach(htmlPlayerEmoji => {
        const isSelected = OptionsUI.isPlayerSelecting(htmlPlayerEmoji);
        if (isSelected) {
          htmlPlayerEmoji.classList.remove(OptionsUI.SELECTING_CLASS_NAME);
        } else {
          htmlPlayerEmoji.classList.add(OptionsUI.SELECTING_CLASS_NAME);
        }
        htmlPlayerEmoji.dataset.playerEmoji = isSelected
          ? ""
          : OptionsUI.SELECTING_CLASS_NAME;
      });
    }
  
    selectEmoji({ target }) {
      if (target.tagName === OptionsUI.EMOJI_TAG_NAME) {
        const emoji = target.cloneNode();
        // Emoji PNG to SVG
        emoji.classList.replace(
          OptionsUI.EMOJI_CLASS_NAME,
          OptionsUI.EMOJI_SVG_CLASS_NAME
        );
        // Update emoji's player
        this.playerSelecting().innerHTML = emoji.outerHTML;
      }
    }
  
    checkValidInput() {
      const isValidInput = [...this.htmlsPlayerName].every(
        htmlPlayerName => htmlPlayerName.value !== ""
      );
      this.htmlOKButton.disabled = !isValidInput;
    }
  
    getPlayers() {
      return [0, 1].map(i => {
        return new Player(
          i,
          this.htmlsPlayerName[i].value,
          this.htmlsPlayerEmoji[i].firstElementChild
        );
      });
    }
  
    initEvents() {
      [
        [[this.htmlEmojis], "click", this.selectEmoji],
        [this.htmlsPlayerEmoji, "click", this.switchPlayers],
        [this.htmlsPlayerName, "keyup", this.checkValidInput]
      ].forEach(([htmlElements, eventName, handlerFunction]) => {
        htmlElements.forEach(htmlElement => {
          htmlElement.addEventListener(eventName, handlerFunction.bind(this));
        });
      });
    }
  }
  
  class BoardUI extends UI {
    /**
     * @constructor
     * @param {HTMLDivElement} htmlContainer
     * @param {HTMLTableElement} htmlBoard
     * @param {NodeList} htmlsSquare
     * @param {NodeList} htmlsPlayerTurn
     * @param {OptionsUI} optionsUI
     * @param {WinnerUI} winnerUI
     * @param {TicTacToe} [ticTacToe]
     */
    constructor(
      htmlContainer,
      htmlBoard,
      htmlsSquare,
      htmlsPlayerTurn,
      optionsUI,
      winnerUI,
      ticTacToe
    ) {
      super(htmlContainer);
      // Define properties
      this.htmlBoard = htmlBoard;
      this.htmlsSquare = htmlsSquare;
      this.htmlsPlayerTurn = htmlsPlayerTurn;
      this.optionsUI = optionsUI;
      this.winnerUI = winnerUI;
      this.ticTacToe = ticTacToe;
      // Execute necessary functions
      this.initEvents();
    }
  
    static SQUARE_TAG_NAME = "TD";
    static SQUARE_MARKED_CLASS_NAME = "marked";
    static PLAYER_TURN_CLASS_NAME = "turn";
  
    openOptionsScreen() {
      this.hide();
      this.optionsUI.show();
    }
  
    clearSquares() {
      this.htmlsSquare.forEach(htmlSquare => {
        htmlSquare.classList.remove(BoardUI.SQUARE_MARKED_CLASS_NAME);
        htmlSquare.innerHTML = "";
      });
    }
  
    renderPlayersTurn() {
      [this.ticTacToe.player1, this.ticTacToe.player2].forEach((player, i) => {
        this.htmlsPlayerTurn[i].innerHTML = `
          <div>${player.emoji.outerHTML}</div>
          <div>${player.name}</div>
        `;
      });
  
      this.htmlsPlayerTurn[0].classList.add(BoardUI.PLAYER_TURN_CLASS_NAME);
      this.htmlsPlayerTurn[1].classList.remove(BoardUI.PLAYER_TURN_CLASS_NAME);
    }
  
    restartTicTacToe() {
      this.ticTacToe = new TicTacToe(...this.optionsUI.getPlayers());
      this.clearSquares();
      this.renderPlayersTurn();
      this.optionsUI.hide();
      this.winnerUI.hide();
      this.show();
    }
  
    handleBoardClick({ target }) {
      if (
        !this.ticTacToe.state.gameOver &&
        target.tagName === BoardUI.SQUARE_TAG_NAME &&
        !target.classList.contains(BoardUI.SQUARE_MARKED_CLASS_NAME)
      ) {
        this.markSquare(
          target,
          Number(target.dataset.square),
          this.ticTacToe.state.turn.emoji.cloneNode()
        );
        this.switchPlayerTurn();
      }
    }
  
    markSquare(square, position, emoji) {
      // Mark square
      square.classList.add(BoardUI.SQUARE_MARKED_CLASS_NAME);
      square.appendChild(emoji);
      // Play ticTacToe object
      this.ticTacToe.play(position);
      // Announce if winner
      if (this.ticTacToe.state.gameOver) {
        this.winnerUI.announceWinner(this.ticTacToe.state.winner);
      }
    }
  
    switchPlayerTurn() {
      this.htmlsPlayerTurn.forEach(htmlPlayerTurn => {
        const className = BoardUI.PLAYER_TURN_CLASS_NAME;
        htmlPlayerTurn.classList.contains(className)
          ? htmlPlayerTurn.classList.remove(className)
          : htmlPlayerTurn.classList.add(className);
      });
    }
  
    initEvents() {
      this.htmlBoard.addEventListener("click", this.handleBoardClick.bind(this));
  
      [this.optionsUI.htmlOKButton, this.winnerUI.htmlRestartButton].forEach(
        htmlElement => {
          htmlElement.addEventListener("click", this.restartTicTacToe.bind(this));
        }
      );
  
      this.htmlsPlayerTurn.forEach(htmlPlayerTurn => {
        htmlPlayerTurn.addEventListener(
          "click",
          this.openOptionsScreen.bind(this)
        );
      });
    }
  }
  
  class WinnerUI extends UI {
    /**
     * @constructor
     * @param {HTMLDivElement} htmlContainer
     * @param {HTMLDivElement} htmlWinnerEmoji
     * @param {HTMLDivElement} htmlMessage
     * @param {HTMLButtonElement} htmlRestartButton
     */
    constructor(htmlContainer, htmlWinnerEmoji, htmlMessage, htmlRestartButton) {
      super(htmlContainer);
      this.htmlWinnerEmoji = htmlWinnerEmoji;
      this.htmlMessage = htmlMessage;
      this.htmlRestartButton = htmlRestartButton;
    }
  
    /**
     *
     * @param {Player} winner
     */
    announceWinner(winner) {
      if (winner) {
        this.htmlWinnerEmoji.innerHTML = `${winner.emoji.outerHTML}`;
        this.htmlMessage.innerHTML = `The winner is ${winner.name}!`;
      } else {
        this.htmlWinnerEmoji.innerHTML = "";
        this.htmlMessage.innerHTML = "It's tie!";
      }
      this.show();
    }
  }
  
  const optionsUI = new OptionsUI(
    document.getElementById("options-screen"),
    document.querySelectorAll("[data-player-name]"),
    document.querySelectorAll("[data-player-emoji]"),
    document.getElementById("emojis"),
    document.getElementById("accept-options")
  );
  
  const winnerUI = new WinnerUI(
    document.getElementById("winning-container"),
    document.getElementById("winner-emoji"),
    document.getElementById("winning-message"),
    document.getElementById("restart-button")
  );
  
  const boardUI = new BoardUI(
    document.getElementById("board-container"),
    document.getElementById("board"),
    document.querySelectorAll("[data-square]"),
    document.querySelectorAll("[data-player-turn]"),
    optionsUI,
    winnerUI
  );
  