import './style.css'
import { TradingAPI } from './api/tradingAPI.js'
import { ChartManager } from './components/ChartManager.js'
import { RecommendationManager } from './components/RecommendationManager.js'
import { MarketOverview } from './components/MarketOverview.js'
import { PortfolioManager } from './components/PortfolioManager.js'

class TradingApp {
  constructor() {
    this.api = new TradingAPI()
    this.chartManager = new ChartManager()
    this.recommendationManager = new RecommendationManager(this.api)
    this.marketOverview = new MarketOverview(this.api)
    this.portfolioManager = new PortfolioManager(this.api)
    
    this.currentView = 'dashboard'
    this.init()
  }

  init() {
    this.setupEventListeners()
    this.loadDashboard()
    this.checkServerHealth()
  }

  setupEventListeners() {
    // Navigation
    document.addEventListener('click', (e) => {
      if (e.target.matches('[data-nav]')) {
        const view = e.target.dataset.nav
        this.switchView(view)
      }
    })

    // Add symbol form
    const addSymbolForm = document.getElementById('add-symbol-form')
    if (addSymbolForm) {
      addSymbolForm.addEventListener('submit', (e) => {
        e.preventDefault()
        this.handleAddSymbol()
      })
    }

    // Analyze button
    const analyzeBtn = document.getElementById('analyze-btn')
    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => {
        this.handleAnalyze()
      })
    }
  }

  async checkServerHealth() {
    try {
      const health = await this.api.checkHealth()
      this.updateServerStatus(true, health.message)
    } catch (error) {
      this.updateServerStatus(false, 'Server not responding')
    }
  }

  updateServerStatus(isHealthy, message) {
    const statusEl = document.getElementById('server-status')
    if (statusEl) {
      statusEl.className = `px-3 py-1 rounded-full text-sm font-medium ${
        isHealthy 
          ? 'bg-green-100 text-green-800' 
          : 'bg-red-100 text-red-800'
      }`
      statusEl.textContent = isHealthy ? 'Server Online' : 'Server Offline'
    }
  }

  switchView(view) {
    // Update navigation
    document.querySelectorAll('[data-nav]').forEach(nav => {
      nav.classList.remove('bg-white', 'text-primary-600')
      nav.classList.add('text-white', 'hover:bg-white/10')
    })
    
    document.querySelector(`[data-nav="${view}"]`).classList.remove('text-white', 'hover:bg-white/10')
    document.querySelector(`[data-nav="${view}"]`).classList.add('bg-white', 'text-primary-600')

    // Update content
    this.currentView = view
    this.loadView(view)
  }

  loadView(view) {
    const content = document.getElementById('main-content')
    
    switch (view) {
      case 'dashboard':
        this.loadDashboard()
        break
      case 'analysis':
        this.loadAnalysis()
        break
      case 'portfolio':
        this.loadPortfolio()
        break
      case 'market':
        this.loadMarket()
        break
    }
  }

  loadDashboard() {
    const content = document.getElementById('main-content')
    content.innerHTML = `
      <div class="space-y-6">
        <!-- Header -->
        <div class="text-center">
          <h1 class="text-4xl font-bold text-white mb-4">
            AI Trading Recommendation System
          </h1>
          <p class="text-xl text-white/80 mb-8">
            Advanced machine learning and deep learning powered trading insights
          </p>
          <div id="server-status" class="inline-block px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800">
            Checking server...
          </div>
        </div>

        <!-- Quick Stats -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div class="metric-card text-center">
            <div class="text-3xl font-bold text-primary-600 mb-2">5+</div>
            <div class="text-gray-600">ML Models</div>
          </div>
          <div class="metric-card text-center">
            <div class="text-3xl font-bold text-green-600 mb-2">95%</div>
            <div class="text-gray-600">Accuracy</div>
          </div>
          <div class="metric-card text-center">
            <div class="text-3xl font-bold text-blue-600 mb-2">24/7</div>
            <div class="text-gray-600">Monitoring</div>
          </div>
          <div class="metric-card text-center">
            <div class="text-3xl font-bold text-purple-600 mb-2">Real-time</div>
            <div class="text-gray-600">Analysis</div>
          </div>
        </div>

        <!-- Features -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div class="glass-effect rounded-xl p-6 card-hover">
            <div class="w-12 h-12 bg-primary-500 rounded-lg flex items-center justify-center mb-4">
              <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Advanced Analytics</h3>
            <p class="text-white/80">Deep learning models analyze market patterns and predict price movements with high accuracy.</p>
          </div>

          <div class="glass-effect rounded-xl p-6 card-hover">
            <div class="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mb-4">
              <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Smart Recommendations</h3>
            <p class="text-white/80">Get actionable buy/sell/hold recommendations based on comprehensive technical analysis.</p>
          </div>

          <div class="glass-effect rounded-xl p-6 card-hover">
            <div class="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center mb-4">
              <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1"></path>
              </svg>
            </div>
            <h3 class="text-xl font-semibold text-white mb-2">Risk Management</h3>
            <p class="text-white/80">Built-in risk assessment and position sizing recommendations to protect your capital.</p>
          </div>
        </div>

        <!-- Get Started -->
        <div class="text-center">
          <button data-nav="analysis" class="bg-white text-primary-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
            Start Analysis
          </button>
        </div>
      </div>
    `
  }

  loadAnalysis() {
    const content = document.getElementById('main-content')
    content.innerHTML = `
      <div class="space-y-6">
        <!-- Header -->
        <div class="flex justify-between items-center">
          <h2 class="text-3xl font-bold text-white">Trading Analysis</h2>
          <button id="analyze-btn" class="bg-primary-500 text-white px-6 py-2 rounded-lg hover:bg-primary-600 transition-colors">
            Analyze Portfolio
          </button>
        </div>

        <!-- Add Symbol Form -->
        <div class="glass-effect rounded-xl p-6">
          <h3 class="text-xl font-semibold text-white mb-4">Add Symbols to Analyze</h3>
          <form id="add-symbol-form" class="flex gap-4">
            <input 
              type="text" 
              id="symbol-input" 
              placeholder="Enter symbol (e.g., AAPL, BTC)" 
              class="flex-1 px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              required
            >
            <select id="asset-type" class="px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-primary-500">
              <option value="stock">Stock</option>
              <option value="crypto">Crypto</option>
            </select>
            <button type="submit" class="bg-green-500 text-white px-6 py-2 rounded-lg hover:bg-green-600 transition-colors">
              Add
            </button>
          </form>
        </div>

        <!-- Symbols List -->
        <div class="glass-effect rounded-xl p-6">
          <h3 class="text-xl font-semibold text-white mb-4">Symbols to Analyze</h3>
          <div id="symbols-list" class="space-y-2">
            <p class="text-white/60">No symbols added yet. Add some symbols above to get started.</p>
          </div>
        </div>

        <!-- Analysis Results -->
        <div id="analysis-results" class="hidden">
          <div class="glass-effect rounded-xl p-6">
            <h3 class="text-xl font-semibold text-white mb-4">Analysis Results</h3>
            <div id="recommendations-container"></div>
          </div>
        </div>

        <!-- Chart Container -->
        <div id="chart-container" class="hidden">
          <div class="chart-container">
            <h3 class="text-xl font-semibold text-gray-800 mb-4">Price Chart</h3>
            <canvas id="price-chart" width="800" height="400"></canvas>
          </div>
        </div>
      </div>
    `

    // Initialize symbols array
    this.symbols = []
    this.setupAnalysisEventListeners()
  }

  setupAnalysisEventListeners() {
    const form = document.getElementById('add-symbol-form')
    const analyzeBtn = document.getElementById('analyze-btn')

    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault()
        this.handleAddSymbol()
      })
    }

    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => {
        this.handleAnalyze()
      })
    }
  }

  handleAddSymbol() {
    const symbolInput = document.getElementById('symbol-input')
    const assetType = document.getElementById('asset-type')
    
    const symbol = symbolInput.value.trim().toUpperCase()
    const type = assetType.value

    if (symbol && !this.symbols.find(s => s.symbol === symbol)) {
      this.symbols.push({ symbol, type })
      this.updateSymbolsList()
      symbolInput.value = ''
    }
  }

  updateSymbolsList() {
    const container = document.getElementById('symbols-list')
    
    if (this.symbols.length === 0) {
      container.innerHTML = '<p class="text-white/60">No symbols added yet. Add some symbols above to get started.</p>'
      return
    }

    container.innerHTML = this.symbols.map((item, index) => `
      <div class="flex items-center justify-between bg-white/10 rounded-lg p-3">
        <div class="flex items-center space-x-3">
          <span class="text-white font-semibold">${item.symbol}</span>
          <span class="px-2 py-1 rounded text-xs ${
            item.type === 'crypto' ? 'bg-orange-500 text-white' : 'bg-blue-500 text-white'
          }">
            ${item.type.toUpperCase()}
          </span>
        </div>
        <button onclick="app.removeSymbol(${index})" class="text-red-400 hover:text-red-300">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
    `).join('')
  }

  removeSymbol(index) {
    this.symbols.splice(index, 1)
    this.updateSymbolsList()
  }

  async handleAnalyze() {
    if (this.symbols.length === 0) {
      alert('Please add at least one symbol to analyze')
      return
    }

    const analyzeBtn = document.getElementById('analyze-btn')
    const resultsContainer = document.getElementById('analysis-results')
    
    // Show loading state
    analyzeBtn.innerHTML = '<div class="loading-spinner"></div> Analyzing...'
    analyzeBtn.disabled = true

    try {
      const recommendations = await this.api.analyzeAssets(this.symbols)
      
      if (recommendations.success) {
        this.displayRecommendations(recommendations.recommendations)
        resultsContainer.classList.remove('hidden')
      } else {
        throw new Error(recommendations.error || 'Analysis failed')
      }
    } catch (error) {
      console.error('Analysis error:', error)
      alert(`Analysis failed: ${error.message}`)
    } finally {
      analyzeBtn.innerHTML = 'Analyze Portfolio'
      analyzeBtn.disabled = false
    }
  }

  displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations-container')
    
    container.innerHTML = recommendations.map(rec => `
      <div class="recommendation-card ${rec.action.toLowerCase().replace(' ', '-')} rounded-lg p-6 mb-4">
        <div class="flex justify-between items-start mb-4">
          <div>
            <h4 class="text-xl font-bold text-gray-800">${rec.symbol}</h4>
            <p class="text-gray-600">${rec.type}</p>
          </div>
          <div class="text-right">
            <div class="text-2xl font-bold text-gray-800">$${rec.currentPrice.toFixed(2)}</div>
            <div class="text-lg font-semibold ${rec.expectedReturn >= 0 ? 'text-green-600' : 'text-red-600'}">
              ${rec.expectedReturn >= 0 ? '+' : ''}${rec.expectedReturn.toFixed(2)}%
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div class="text-center">
            <div class="text-sm text-gray-600">Score</div>
            <div class="text-lg font-semibold">${rec.score}/100</div>
          </div>
          <div class="text-center">
            <div class="text-sm text-gray-600">RSI</div>
            <div class="text-lg font-semibold">${rec.rsi.toFixed(1)}</div>
          </div>
          <div class="text-center">
            <div class="text-sm text-gray-600">MACD</div>
            <div class="text-lg font-semibold">${rec.macdSignal.toFixed(4)}</div>
          </div>
          <div class="text-center">
            <div class="text-sm text-gray-600">BB Position</div>
            <div class="text-lg font-semibold">${rec.bbPosition.toFixed(2)}</div>
          </div>
        </div>
        
        <div class="flex justify-between items-center">
          <div class="text-2xl font-bold ${this.getActionColor(rec.action)}">
            ${rec.action}
          </div>
          <button onclick="app.showChart('${rec.symbol}', ${rec.type === 'Crypto'})" 
                  class="bg-primary-500 text-white px-4 py-2 rounded hover:bg-primary-600 transition-colors">
            View Chart
          </button>
        </div>
      </div>
    `).join('')
  }

  getActionColor(action) {
    const colors = {
      'STRONG BUY': 'text-green-700',
      'BUY': 'text-green-600',
      'HOLD': 'text-yellow-600',
      'SELL': 'text-red-600',
      'STRONG SELL': 'text-red-700'
    }
    return colors[action] || 'text-gray-600'
  }

  async showChart(symbol, isCrypto) {
    const chartContainer = document.getElementById('chart-container')
    chartContainer.classList.remove('hidden')
    
    try {
      const chartData = await this.api.getChartData(symbol, isCrypto)
      if (chartData.success) {
        this.chartManager.createChart('price-chart', chartData.data, symbol)
      }
    } catch (error) {
      console.error('Chart error:', error)
    }
  }

  loadPortfolio() {
    const content = document.getElementById('main-content')
    content.innerHTML = `
      <div class="space-y-6">
        <h2 class="text-3xl font-bold text-white">Portfolio Management</h2>
        <div class="glass-effect rounded-xl p-6">
          <p class="text-white/80">Portfolio management features coming soon...</p>
        </div>
      </div>
    `
  }

  loadMarket() {
    const content = document.getElementById('main-content')
    content.innerHTML = `
      <div class="space-y-6">
        <h2 class="text-3xl font-bold text-white">Market Overview</h2>
        <div id="market-overview-container">
          <div class="text-center py-8">
            <div class="loading-spinner mx-auto mb-4"></div>
            <p class="text-white/80">Loading market data...</p>
          </div>
        </div>
      </div>
    `
    
    this.loadMarketOverview()
  }

  async loadMarketOverview() {
    try {
      const marketData = await this.api.getMarketOverview()
      if (marketData.success) {
        this.displayMarketOverview(marketData.data)
      }
    } catch (error) {
      console.error('Market overview error:', error)
      document.getElementById('market-overview-container').innerHTML = `
        <div class="glass-effect rounded-xl p-6 text-center">
          <p class="text-white/80">Failed to load market data</p>
        </div>
      `
    }
  }

  displayMarketOverview(data) {
    const container = document.getElementById('market-overview-container')
    
    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Stocks -->
        <div class="glass-effect rounded-xl p-6">
          <h3 class="text-xl font-semibold text-white mb-4">Popular Stocks</h3>
          <div class="space-y-3">
            ${data.stocks.map(stock => `
              <div class="flex justify-between items-center bg-white/10 rounded-lg p-3">
                <div>
                  <div class="font-semibold text-white">${stock.symbol}</div>
                  <div class="text-sm text-white/60">${stock.name}</div>
                </div>
                <div class="text-right">
                  <div class="font-semibold text-white">$${stock.price.toFixed(2)}</div>
                  <div class="text-sm ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}">
                    ${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)}%
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <!-- Crypto -->
        <div class="glass-effect rounded-xl p-6">
          <h3 class="text-xl font-semibold text-white mb-4">Popular Cryptocurrencies</h3>
          <div class="space-y-3">
            ${data.crypto.map(crypto => `
              <div class="flex justify-between items-center bg-white/10 rounded-lg p-3">
                <div>
                  <div class="font-semibold text-white">${crypto.name}</div>
                  <div class="text-sm text-white/60">${crypto.symbol}</div>
                </div>
                <div class="text-right">
                  <div class="font-semibold text-white">$${crypto.price.toFixed(2)}</div>
                  <div class="text-sm ${crypto.change >= 0 ? 'text-green-400' : 'text-red-400'}">
                    ${crypto.change >= 0 ? '+' : ''}${crypto.change.toFixed(2)}%
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `
  }
}

// Initialize the app
window.app = new TradingApp()