import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { GoogleOAuthProvider } from '@react-oauth/google';
import './App.css';
import Home from "./components/Home";
import Viz from "./components/Viz";
import VizSample from "./components/VizSample";
import Welcome from "./components/Welcome";
import Performance from "./components/Performance";
import Algorithm from "./components/Algorithm";

const App = () => {
  return (
    <GoogleOAuthProvider clientId='132019401240-5mgdpd88e5ro256pofe8r1p9srqo518p.apps.googleusercontent.com'>
      <Router>
        <div>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/Viz" element={<Viz />} />
              <Route path="/Welcome" element={<Welcome />} />
              <Route path="/VizSample" element={<VizSample />} />
              <Route path="/Performance" element={<Performance />} />
              <Route path="/Algorithm" element={<Algorithm />} />
            </Routes>
        </div>
      </Router>
    </GoogleOAuthProvider> 
  );
};

export default App;