import React from "react";
import Operator from "./Operator";
class App extends React.Component{
  constructor(props){
    super(props);
    this.state={count:0};

  this.modifyState=this.modifyState.bind(this);
}

componentDidMount(){
  this.setState({coount:0});
}
modifyState(newCount){
  this.setState({count:newCount});
}
render(){
  return(
    <div style={{textAlign:"center"}}>
      <h1>Counter:{this.state.count}</h1>
      <Operator count={this.state.count} modifyState={this.modifyState}/>
    </div>
  );
}
}
export default App;