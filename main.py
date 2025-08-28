"""
Advanced Stock & Cryptocurrency Trading Recommendation System
Using Machine Learning and Deep Learning Techniques
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam

# Technical Analysis
import ta
from ta import add_all_ta_features
from ta.volatility import bollinger_hband, average_true_range
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MarketDataCollector:
    """Collect and preprocess market data for stocks and cryptocurrencies"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def fetch_data(self, symbol, period='2y', interval='1d', crypto=False):
        """
        Fetch historical data for a given symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add symbol column
            data['Symbol'] = symbol
            data['Asset_Type'] = 'Crypto' if crypto else 'Stock'
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        """
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Price'] = df['Volume'] * df['Close']
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'EMA_{period}'] = EMAIndicator(df['Close'], window=period).ema_indicator()
            
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # RSI
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = RSIIndicator(df['Close'], window=period).rsi()
        
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # ADX
        adx = ADXIndicator(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()
        
        # ATR (Volatility)
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
        df['ATR'] = atr.average_true_range()
        
        # Support and Resistance Levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Support_Resistance_Ratio'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        return df
    
    def create_features(self, df, lookback=60):
        """
        Create features for ML models including lag features
        """
        # Lag features
        for i in range(1, lookback + 1):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            if i <= 10:
                df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
                df[f'Returns_Lag_{i}'] = df['Returns'].shift(i)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
            
        # Price patterns
        df['Higher_High'] = ((df['High'] > df['High'].shift(1)) & 
                            (df['High'].shift(1) > df['High'].shift(2))).astype(int)
        df['Lower_Low'] = ((df['Low'] < df['Low'].shift(1)) & 
                          (df['Low'].shift(1) < df['Low'].shift(2))).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df

class MLTradingModels:
    """Machine Learning models for trading predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self, df, target_days=5, test_size=0.2):
        """
        Prepare data for training
        """
        # Create target variable (future returns)
        df[f'Target_{target_days}d'] = df['Close'].shift(-target_days) / df['Close'] - 1
        
        # Remove last rows with NaN targets
        df = df[:-target_days]
        
        # Select features
        feature_cols = [col for col in df.columns if col not in 
                       ['Symbol', 'Asset_Type', f'Target_{target_days}d', 
                        'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = df[feature_cols].values
        y = df[f'Target_{target_days}d'].values
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest model
        """
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        print(f"Random Forest - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        return model, test_pred
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model
        """
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        print(f"XGBoost - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        return model, test_pred
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Train Gradient Boosting model
        """
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        print(f"Gradient Boosting - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        return model, test_pred

class DLTradingModels:
    """Deep Learning models for trading predictions"""
    
    def __init__(self):
        self.models = {}
        self.history = {}
        
    def prepare_sequences(self, df, sequence_length=60, target_days=5):
        """
        Prepare sequences for LSTM/GRU models
        """
        # Select features for sequences
        feature_cols = ['Close', 'Volume', 'Returns', 'RSI_14', 'MACD', 
                       'BB_Position', 'Support_Resistance_Ratio']
        
        # Ensure features exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        data = df[feature_cols].values
        
        # Scale data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data_scaled) - target_days):
            X.append(data_scaled[i-sequence_length:i])
            y.append(data_scaled[i + target_days, 0])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM model
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_gru_model(self, input_shape):
        """
        Build GRU model
        """
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def build_transformer_model(self, input_shape):
        """
        Build Transformer-based model for time series
        """
        inputs = Input(shape=input_shape)
        
        # Position encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [tf.shape(inputs)[0], 1])
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=32
        )(inputs, inputs)
        attention_output = Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward network
        ffn_output = Dense(64, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[1])(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        # Global pooling and output
        pooled = layers.GlobalAveragePooling1D()(ffn_output)
        outputs = Dense(32, activation='relu')(pooled)
        outputs = Dropout(0.1)(outputs)
        outputs = Dense(1)(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name='Model'):
        """
        Train deep learning model
        """
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"{model_name} - Train Loss: {train_loss[0]:.6f}, Test Loss: {test_loss[0]:.6f}")
        
        return model, history

class TradingRecommendationSystem:
    """Main system for generating trading recommendations"""
    
    def __init__(self):
        self.data_collector = MarketDataCollector()
        self.ml_models = MLTradingModels()
        self.dl_models = DLTradingModels()
        self.recommendations = []
        
    def analyze_asset(self, symbol, crypto=False):
        """
        Comprehensive analysis of a single asset
        """
        print(f"\nAnalyzing {symbol}...")
        
        # Fetch data
        df = self.data_collector.fetch_data(symbol, crypto=crypto)
        if df is None:
            return None
        
        # Calculate indicators
        df = self.data_collector.calculate_technical_indicators(df)
        df = self.data_collector.create_features(df)
        
        # ML predictions
        X_train, X_test, y_train, y_test, scaler, features = self.ml_models.prepare_data(df)
        
        rf_model, rf_pred = self.ml_models.train_random_forest(X_train, y_train, X_test, y_test)
        xgb_model, xgb_pred = self.ml_models.train_xgboost(X_train, y_train, X_test, y_test)
        gb_model, gb_pred = self.ml_models.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # DL predictions
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, seq_scaler = self.dl_models.prepare_sequences(df)
        
        lstm_model = self.dl_models.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        lstm_model, lstm_history = self.dl_models.train_model(
            lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, "LSTM"
        )
        
        gru_model = self.dl_models.build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))
        gru_model, gru_history = self.dl_models.train_model(
            gru_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, "GRU"
        )
        
        # Ensemble predictions
        ml_ensemble = (rf_pred + xgb_pred + gb_pred) / 3
        lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
        gru_pred = gru_model.predict(X_test_seq, verbose=0).flatten()
        
        # Calculate signals
        current_price = df['Close'].iloc[-1]
        avg_ml_prediction = np.mean(ml_ensemble[-5:])  # Last 5 predictions
        avg_dl_prediction = np.mean([lstm_pred[-1], gru_pred[-1]])
        
        # Technical signals
        rsi = df['RSI_14'].iloc[-1]
        macd_signal = df['MACD_Diff'].iloc[-1]
        bb_position = df['BB_Position'].iloc[-1]
        
        # Recommendation score (0-100)
        score = 50  # Neutral start
        
        # ML/DL signals
        if avg_ml_prediction > 0.02:
            score += 15
        elif avg_ml_prediction < -0.02:
            score -= 15
            
        # Technical signals
        if rsi < 30:
            score += 10  # Oversold
        elif rsi > 70:
            score -= 10  # Overbought
            
        if macd_signal > 0:
            score += 5
        else:
            score -= 5
            
        if bb_position < 0.2:
            score += 10  # Near lower band
        elif bb_position > 0.8:
            score -= 10  # Near upper band
            
        # Trend analysis
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        if current_price > sma_20 > sma_50:
            score += 10  # Uptrend
        elif current_price < sma_20 < sma_50:
            score -= 10  # Downtrend
            
        # Generate recommendation
        if score >= 70:
            action = "STRONG BUY"
        elif score >= 60:
            action = "BUY"
        elif score >= 40:
            action = "HOLD"
        elif score >= 30:
            action = "SELL"
        else:
            action = "STRONG SELL"
            
        recommendation = {
            'Symbol': symbol,
            'Type': 'Crypto' if crypto else 'Stock',
            'Current_Price': current_price,
            'Score': score,
            'Action': action,
            'ML_Prediction': avg_ml_prediction,
            'RSI': rsi,
            'MACD_Signal': macd_signal,
            'BB_Position': bb_position,
            'Feature_Importance': dict(zip(features[:10], rf_model.feature_importances_[:10]))
        }
        
        return recommendation
    
    def analyze_portfolio(self, symbols_dict):
        """
        Analyze multiple assets
        symbols_dict: {'stocks': [...], 'crypto': [...]}
        """
        all_recommendations = []
        
        # Analyze stocks
        if 'stocks' in symbols_dict:
            for symbol in symbols_dict['stocks']:
                rec = self.analyze_asset(symbol, crypto=False)
                if rec:
                    all_recommendations.append(rec)
                    
        # Analyze crypto
        if 'crypto' in symbols_dict:
            for symbol in symbols_dict['crypto']:
                rec = self.analyze_asset(symbol, crypto=True)
                if rec:
                    all_recommendations.append(rec)
                    
        # Sort by score
        all_recommendations.sort(key=lambda x: x['Score'], reverse=True)
        
        return all_recommendations
    
    def generate_report(self, recommendations):
        """
        Generate detailed report
        """
        print("\n" + "="*80)
        print("TRADING RECOMMENDATIONS REPORT")
        print("="*80)
        
        for rec in recommendations:
            print(f"\n{rec['Symbol']} ({rec['Type']})")
            print("-" * 40)
            print(f"Current Price: ${rec['Current_Price']:.2f}")
            print(f"Recommendation Score: {rec['Score']}/100")
            print(f"ACTION: {rec['Action']}")
            print(f"Expected Return (ML): {rec['ML_Prediction']*100:.2f}%")
            print(f"RSI: {rec['RSI']:.2f}")
            print(f"MACD Signal: {rec['MACD_Signal']:.4f}")
            print(f"Bollinger Band Position: {rec['BB_Position']:.2f}")
            
            print("\nTop Features Influencing Prediction:")
            for feat, importance in sorted(rec['Feature_Importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {feat}: {importance:.4f}")
    
    def plot_analysis(self, symbol, crypto=False):
        """
        Create visualization of analysis
        """
        # Fetch fresh data for plotting
        df = self.data_collector.fetch_data(symbol, period='6mo', crypto=crypto)
        df = self.data_collector.calculate_technical_indicators(df)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(f'{symbol} Price & Moving Averages', 
                           'Volume', 'RSI', 'MACD')
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                          low=df['Low'], close=df['Close'], name='Price'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='gray', width=0.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='gray', width=0.5), fill='tonexty'),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume'),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], name='RSI',
                      line=dict(color='purple', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='blue', width=1)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                      line=dict(color='red', width=1)),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            height=1000,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update x-axis
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

# Main execution example
def main():
    """
    Example usage of the trading system
    """
    # Initialize system
    system = TradingRecommendationSystem()
    
    # Define portfolio
    portfolio = {
        'stocks': ['AAPL'],
    }
    
    print("Starting Advanced Trading Analysis System...")
    print("This may take several minutes to complete all analyses...")
    
    # Analyze portfolio
    recommendations = system.analyze_portfolio(portfolio)
    
    # Generate report
    system.generate_report(recommendations)
    
    # Show top recommendations
    print("\n" + "="*80)
    print("TOP 3 RECOMMENDATIONS")
    print("="*80)
    
    for rec in recommendations[:3]:
        print(f"\n{rec['Symbol']}: {rec['Action']} (Score: {rec['Score']}/100)")
        print(f"  Expected Return: {rec['ML_Prediction']*100:.2f}%")
    
    # Create visualization for top pick
    if recommendations:
        top_pick = recommendations[0]
        print(f"\nGenerating detailed chart for top pick: {top_pick['Symbol']}")
        fig = system.plot_analysis(top_pick['Symbol'], 
                                  crypto=(top_pick['Type']=='Crypto'))
        fig.show()
    
    return recommendations

# Risk Management Module
class RiskManager:
    """
    Risk management and position sizing
    """
    
    @staticmethod
    def calculate_position_size(portfolio_value, risk_per_trade=0.02, 
                               stop_loss_pct=0.05):
        """
        Calculate optimal position size based on Kelly Criterion
        """
        risk_amount = portfolio_value * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return min(position_size, portfolio_value * 0.25)  # Max 25% per position
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.95):
        """
        Calculate Value at Risk
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Calculate Sharpe Ratio
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

if __name__ == "__main__":
    # Run the system
    recommendations = main()
    
    print("\n" + "="*80)
    print("IMPORTANT DISCLAIMERS:")
    print("="*80)
    print("1. This is for educational purposes only - not financial advice")
    print("2. Past performance does not guarantee future results")
    print("3. Always do your own research before trading")
    print("4. Consider consulting with a financial advisor")
    print("5. Never invest more than you can afford to lose")
    print("="*80)