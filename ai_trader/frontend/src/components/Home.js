import { BrowserRouter as Router, Route, Routes, useNavigate } from "react-router-dom";
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import './App.css';
import { withRouter } from 'react-router-dom';
import axios from 'axios';
import { jwtDecode } from "jwt-decode";



const Home = ({ history, sendData}) => {
    const navigate = useNavigate();

    const responseGoogleLogin = async (credentialResponse) => {
        console.log('Google Sign-In Success:', credentialResponse);
        const decoded = jwtDecode(credentialResponse.credential);
        console.log(decoded)


        try {
            const response = await axios.post('http://127.0.0.1:5000/verify_sign_up', {
              "email": decoded.email,
            });
            console.log(response)
            const result = response.data

            navigate('/Welcome', {
                state: {
                    email: decoded.email,
                    name: decoded.name,
                }
            })
          
        } catch (error) {
            console.error('Error getting role:', error);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
            <h1>Welcome to AI Lion Trader!</h1>
            <h2>Login to Continue: </h2>
            <GoogleLogin
                onSuccess={responseGoogleLogin}
                shape="circle"
                size="large"
                theme="filled_blue"
                onError={() => {
                console.log('Login Failed');
                }}
            />
            </header>
        </div>
  );
};

export default Home;