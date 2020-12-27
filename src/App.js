import React, { Component } from "react";
import Routes from "./Routes";
import NavBar from "./components/navbar";
import "./App.css";

class App extends Component {
  render() {
    return (
      <React.Fragment>
        <NavBar />
        <Routes />
      </React.Fragment>
    );
  }
}

export default App;
