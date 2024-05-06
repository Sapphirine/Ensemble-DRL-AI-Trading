import { BrowserRouter as Router, Route, Routes, useNavigate, useLocation } from "react-router-dom";
import './App.css';
import { withRouter } from 'react-router-dom';
import axios from 'axios';
import React, { useEffect } from "react";
import * as d3 from "d3";
import { createContext, useContext, useState } from 'react';
import Button from './Button';
import Button2 from './Button2';



const Welcome = (props) => {
    const navigate = useNavigate();
    const location = useLocation();
    const queryParams = new URLSearchParams(location.state);
    let email = queryParams.get('email');
    let name = queryParams.get('name');
    console.log(queryParams)
    const [riskLevel, setRiskLevel] = useState([])
    const [initialRender, setInitialRender] = useState(true);


    useEffect(() =>{
        const fetchData = async(e) => {
          try {
            const response = await axios.post('http://127.0.0.1:5000/get_risk', {
                "email": email,
            });
            
            const result = response.data
            setRiskLevel(result['risk_level'])
            console.log(result['message'])
      
          } catch (error) {
              console.error('Error during POST get_risk:', error);
          }
        }
        fetchData();
    },[]);

    const handleSubmitLow = () => {
        setRiskLevel('low')
    };

    const handleSubmitMed = () => {
        setRiskLevel('medium')
    };

    const handleSubmitHigh = () => {
        setRiskLevel('high')
    };

    const handlePO = async () => {
        console.log('clicked po')
        navigate('/Performance')
    };
    
    const handleAP = async () => {
        console.log('clicked ap')
        navigate('/Algorithm')
    };

    useEffect(() => {
        if (!initialRender) {
            console.log('Risk level changed:', riskLevel);
            post_risk_level();
        } else {
            setInitialRender(false);
        }
    }, [riskLevel, initialRender]);

    const handleClickNull = async () => {
        console.log('clicked null')
    };

    const post_risk_level = async(e) => {
        try {
          const response = await axios.post('http://127.0.0.1:5000/set_risk', {
                "email": email,
                "risk_level": riskLevel,
          });

          const result = response.data
          console.log(result['message'])
              
        } catch (error) {
            console.error('Error during GET set_risk:', error);
        }
    }
    
    
    return (
       <div className="App">
            <header className="App-header">
                <h1>Welcome {name}! </h1>
                <sub>User email: {email}</sub>                    
                <div>
                    <h1> </h1>
                    <h2>Select your Basket</h2>
                    <h5>Current Basket: DOW30 </h5>
                    <Button2 label="DOW30" onClick={handleClickNull} ></Button2>
                    <Button2 label="S&P500" onClick={handleClickNull} ></Button2>
                    <Button2 label="HSI" onClick={handleClickNull} ></Button2>
                    <Button2 label="DJIA" onClick={handleClickNull} ></Button2>
                    <Button2 label="SSE50" onClick={handleClickNull} ></Button2>
                    <Button2 label="CSE" onClick={handleClickNull} ></Button2>
                    <h1> </h1>
                </div>
                <div>
                    <h1> </h1>
                    <h2>Set your Risk Level Preference</h2>
                    <h4>Current Risk Level: {riskLevel} </h4>
                </div>
                <div>
                    <Button label="Low" onClick={handleSubmitLow} style={{ backgroundColor: 'deepskyblue' }}></Button>
                    <Button label="Medium" onClick={handleSubmitMed} style={{ backgroundColor: 'blue' }}></Button>
                    <Button label="High" onClick={handleSubmitHigh} style={{ backgroundColor: 'darkblue' }}></Button>
                </div>
                <div>
                    <h1> </h1>
                    <h2>Account Options</h2>
                    <Button label="Performance Overview" onClick={handlePO} ></Button>
                    <Button label="Algorithm Preview" onClick={handleAP} ></Button>
                </div>
            </header>
        </div>
    );
};

export default Welcome;