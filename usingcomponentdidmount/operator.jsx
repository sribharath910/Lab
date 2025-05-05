import React from "react";

class Operator extends React.Component {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick(e) {
    var click = e.target.id;
    var current = this.props.count;

    if (click === "one") {
      this.props.modifyState(current + 1);
    } else if (click === "two") {
      this.props.modifyState(current - 1);
    } else if (click === "three") {
      this.props.modifyState(current * 2);
    } else if (click === "four") {
      this.props.modifyState(0);
    }
  }

  render() {
    return (
      <div>
        <button id="one" onClick={this.handleClick}>Increment</button>{" "}
        <button id="two" onClick={this.handleClick}>Decrement</button>{" "}
        <button id="three" onClick={this.handleClick}>Double</button>{" "}
        <button id="four" onClick={this.handleClick}>Reset</button>
      </div>
    );
  }
}

export default Operator;