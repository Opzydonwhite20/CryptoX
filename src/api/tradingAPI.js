export class TradingAPI {
  constructor() {
    this.baseURL = '/api'
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    }

    try {
      const response = await fetch(url, config)
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`)
      }
      
      return data
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error)
      throw error
    }
  }

  async checkHealth() {
    return this.request('/health')
  }

  async analyzeAssets(symbols) {
    return this.request('/analyze', {
      method: 'POST',
      body: JSON.stringify({ symbols })
    })
  }

  async getChartData(symbol, crypto = false, period = '6mo') {
    return this.request(`/chart-data/${symbol}?crypto=${crypto}&period=${period}`)
  }

  async getMarketOverview() {
    return this.request('/market-overview')
  }

  async getRiskAnalysis(portfolioValue, positions) {
    return this.request('/risk-analysis', {
      method: 'POST',
      body: JSON.stringify({
        portfolioValue,
        positions
      })
    })
  }
}