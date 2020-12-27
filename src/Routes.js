import React from "react";
import { Route, Switch } from "react-router-dom";
import Home from "./components/home";
import AboutMe from "./components/aboutme";
import Projects from "./components/projects";

const Routes = () => {
  return (
    <Switch>
      <Route exact path="/">
        <Home />
      </Route>
      <Route exact path="/projects">
        <Projects />
      </Route>
      <Route exact path="/aboutme">
        <AboutMe />
      </Route>
    </Switch>
  );
};

export default Routes;
