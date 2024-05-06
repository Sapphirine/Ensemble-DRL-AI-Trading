import { BrowserRouter as Router, Route, Routes, useNavigate, useLocation } from "react-router-dom";
import './App.css';
import { withRouter } from 'react-router-dom';
import axios from 'axios';
import React, { useEffect } from "react";
import * as d3 from "d3";
import { createContext, useContext, useState } from 'react';
import Button from './Button';
import rft_90 from './result_final_turbulance_90.png';
import rft_120 from './result_final_turbulance_120.png';
import rft_inf from './result_final_turbulance_inf.png';
import data_json from './data.json'; 
import annualized_90 from './Annualized_90.png';
import annualized_120 from './Annualized_120.png'; 
import annualized_inf from './Annualized_inf.png'; 


const Algorithm = (props) => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.state);
    const [riskLevel, setRiskLevel] = useState([])
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState({});
    const [jsonData, setJsonData] = useState(data_json);

    const [dispTop, setDispTop] = useState(null);
    const [dispBot, setDispBot] = useState(null);
    const [dispA2cTop, setA2cDispTop] = useState(null);
    const [dispA2cBot, setA2cDispBot] = useState(null);
    const [vizGraph, setVizGraph] = useState(null);
    const [annGraph, setAnnGraph] = useState(null);
    const [showDiv, setShowDiv] = useState(false);
    

    const handleSubmitLow = async () => {
        console.log('clicked low')
        setShowDiv(true);
        setRiskLevel('low')
        setDispTop(Object.fromEntries(Object.entries(data_json).slice(0, 1)));
        setDispBot(Object.fromEntries(Object.entries(data_json).slice(1, 2)));
        setA2cDispTop(Object.fromEntries(Object.entries(data_json).slice(6, 7)));
        setA2cDispBot(Object.fromEntries(Object.entries(data_json).slice(7, 8)));
        setVizGraph(rft_90);
        setAnnGraph(annualized_90);
    };

    const handleSubmitMed = async () => {
        console.log('clicked med')
        setShowDiv(true);
        setRiskLevel('medium')
        setDispTop(Object.fromEntries(Object.entries(data_json).slice(2, 3)));
        setDispBot(Object.fromEntries(Object.entries(data_json).slice(3, 4)));
        setA2cDispTop(Object.fromEntries(Object.entries(data_json).slice(8, 9)));
        setA2cDispBot(Object.fromEntries(Object.entries(data_json).slice(9, 10)));
        setVizGraph(rft_120);
        setAnnGraph(annualized_120);
    };

    const handleSubmitHigh = async () => {
        console.log('clicked hi')
        setShowDiv(true);
        setRiskLevel('high')
        setDispTop(Object.fromEntries(Object.entries(data_json).slice(4, 5)));
        setDispBot(Object.fromEntries(Object.entries(data_json).slice(5, 6)));
        setA2cDispTop(Object.fromEntries(Object.entries(data_json).slice(10, 11)));
        setA2cDispBot(Object.fromEntries(Object.entries(data_json).slice(11, 12)));
        setVizGraph(rft_inf)
        setAnnGraph(annualized_inf);
    };

    const handleClear = async () => {
        console.log('clicked hi')
        setShowDiv(false);
        setRiskLevel([])
        setDispTop(null);
        setDispBot(null);
        setA2cDispTop(null);
        setA2cDispBot(null);
        setVizGraph(null)
        setAnnGraph(null);
    };


    
    return (
       <div className="App">
            <header className="App-header">
                <h1>Algorithm Suggestion</h1>
                <h2>Select Risk Level Button for Trade Recommendations </h2>
                <div>
                    <Button label="Low" onClick={handleSubmitLow} style={{ backgroundColor: 'deepskyblue' }}></Button>
                    <Button label="Medium" onClick={handleSubmitMed} style={{ backgroundColor: 'blue' }}></Button>
                    <Button label="High" onClick={handleSubmitHigh} style={{ backgroundColor: 'darkblue' }}></Button>
                    {showDiv && (
                        <div>
                            <h2>Performance Preview</h2>
                            <img src={vizGraph} width="1000" />
                            <h2>Annualized Summary</h2>
                            <img src={annGraph} width="1000"/>
                            <h2>PPO Top Recs</h2>
                            <table style={{ display: 'flex', justifyContent: 'center' }}>
                                <tbody>
                                <tr>
                                    {/* Map over the values of the JSON object and render each value */}
                                    { dispA2cTop && Object.values(dispA2cTop).map((values, index) => (
                                    values.map((value, idx) => (
                                        <td key={idx} style={{ padding: '8px', color: 'green' }}>{value}</td>
                                    ))
                                    ))}
                                </tr>
                                </tbody>
                            </table>

                            <h2>PPO Bottom Recs</h2>
                            <table style={{ display: 'flex', justifyContent: 'center' }}>
                                <tbody>
                                <tr>
                                    {/* Map over the values of the JSON object and render each value */}
                                    { dispA2cBot && Object.values(dispA2cBot).map((values, index) => (
                                    values.map((value, idx) => (
                                        <td key={idx} style={{ padding: '8px', color: 'red'  }}>{value}</td>
                                    ))
                                    ))}
                                </tr>
                                </tbody>
                            </table>

                            <h2>A2C Top Recs</h2>
                                <table style={{ display: 'flex', justifyContent: 'center' }}>
                                    <tbody>
                                    <tr>
                                        {/* Map over the values of the JSON object and render each value */}
                                        { dispTop && Object.values(dispTop).map((values, index) => (
                                        values.map((value, idx) => (
                                            <td key={idx} style={{ padding: '8px', color: 'green' }}>{value}</td>
                                        ))
                                        ))}
                                    </tr>
                                    </tbody>
                            </table>

                            <h2>A2C Bottom Recs</h2>
                            <table style={{ display: 'flex', justifyContent: 'center' }}>
                                <tbody>
                                <tr>
                                    {/* Map over the values of the JSON object and render each value */}
                                    { dispBot && Object.values(dispBot).map((values, index) => (
                                    values.map((value, idx) => (
                                        <td key={idx} style={{ padding: '8px', color: 'red'  }}>{value}</td>
                                    ))
                                    ))}
                                </tr>
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
                <h1> </h1>
                <Button label="Clear" onClick={handleClear} style={{ backgroundColor: 'red' }}></Button>
            </header>
            
        </div>
    );
};

export default Algorithm;