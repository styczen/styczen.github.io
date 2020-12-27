import React, { Component } from "react";
import Counters from "./projects/counters";

class Projects extends Component {
  render() {
    return (
      <div>
        <h1>My projects</h1>
        <h3>Counters</h3>
        <Counters />
      </div>
    );
  }
}

export default Projects;
