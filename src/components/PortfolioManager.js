export class PortfolioManager {
  constructor(api) {
    this.api = api
    this.portfolio = {
      totalValue: 10000,
      positions: [],
      cash: 10000
    }
    this.loadPortfolio()
  }

  loadPortfolio() {
    const saved = localStorage.getItem('trading-portfolio')
    if (saved) {
      this.portfolio = JSON.parse(saved)
    }
  }

  savePortfolio() {
    localStorage.setItem('trading-portfolio', JSON.stringify(this.portfolio))
  }

  addPosition(symbol, type, shares, price) {
    const existingPosition = this.portfolio.positions.find(p => p.symbol === symbol)
    
    if (existingPosition) {
      // Update existing position
      const totalShares = existingPosition.shares + shares
      const totalValue = (existingPosition.shares * existingPosition.avgPrice) + (shares * price)
      existingPosition.avgPrice = totalValue / totalShares
      existingPosition.shares = totalShares
      existingPosition.lastPrice = price
    } else {
      // Add new position
      this.portfolio.positions.push({
        symbol,
        type,
        shares,
        avgPrice: price,
        lastPrice: price,
        addedDate: new Date().toISOString()
      })
    }

    // Update cash
    this.portfolio.cash -= shares * price
    this.updatePortfolioValue()
    this.savePortfolio()
  }

  removePosition(symbol, shares) {
    const position = this.portfolio.positions.find(p => p.symbol === symbol)
    if (!position) return false

    if (shares >= position.shares) {
      // Remove entire position
      this.portfolio.cash += position.shares * position.lastPrice
      this.portfolio.positions = this.portfolio.positions.filter(p => p.symbol !== symbol)
    } else {
      // Partial sale
      this.portfolio.cash += shares * position.lastPrice
      position.shares -= shares
    }

    this.updatePortfolioValue()
    this.savePortfolio()
    return true
  }

  updatePortfolioValue() {
    const positionsValue = this.portfolio.positions.reduce((total, position) => {
      return total + (position.shares * position.lastPrice)
    }, 0)
    
    this.portfolio.totalValue = this.portfolio.cash + positionsValue
  }

  async updatePositionPrices() {
    // This would typically fetch current prices for all positions
    // For now, we'll simulate this
    for (const position of this.portfolio.positions) {
      try {
        // In a real implementation, you'd fetch current price here
        // position.lastPrice = await this.getCurrentPrice(position.symbol)
      } catch (error) {
        console.error(`Failed to update price for ${position.symbol}:`, error)
      }
    }
    
    this.updatePortfolioValue()
    this.savePortfolio()
  }

  getPortfolioSummary() {
    const totalInvested = this.portfolio.positions.reduce((total, position) => {
      return total + (position.shares * position.avgPrice)
    }, 0)

    const currentValue = this.portfolio.positions.reduce((total, position) => {
      return total + (position.shares * position.lastPrice)
    }, 0)

    const totalGainLoss = currentValue - totalInvested
    const totalGainLossPercent = totalInvested > 0 ? (totalGainLoss / totalInvested) * 100 : 0

    return {
      totalValue: this.portfolio.totalValue,
      cash: this.portfolio.cash,
      invested: totalInvested,
      currentValue,
      gainLoss: totalGainLoss,
      gainLossPercent: totalGainLossPercent,
      positionCount: this.portfolio.positions.length
    }
  }

  renderPortfolio(container) {
    const summary = this.getPortfolioSummary()

    container.innerHTML = `
      <div class="space-y-6">
        <!-- Portfolio Summary -->
        <div class="glass-effect rounded-xl p-6">
          <h3 class="text-xl font-semibold text-white mb-4">Portfolio Summary</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="text-center">
              <div class="text-2xl font-bold text-white">$${summary.totalValue.toFixed(2)}</div>
              <div class="text-sm text-white/60">Total Value</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold text-white">$${summary.cash.toFixed(2)}</div>
              <div class="text-sm text-white/60">Cash</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold ${summary.gainLoss >= 0 ? 'text-green-400' : 'text-red-400'}">
                ${summary.gainLoss >= 0 ? '+' : ''}$${summary.gainLoss.toFixed(2)}
              </div>
              <div class="text-sm text-white/60">Gain/Loss</div>
            </div>
            <div class="text-center">
              <div class="text-2xl font-bold ${summary.gainLossPercent >= 0 ? 'text-green-400' : 'text-red-400'}">
                ${summary.gainLossPercent >= 0 ? '+' : ''}${summary.gainLossPercent.toFixed(2)}%
              </div>
              <div class="text-sm text-white/60">Return</div>
            </div>
          </div>
        </div>

        <!-- Positions -->
        <div class="glass-effect rounded-xl p-6">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-semibold text-white">Positions</h3>
            <button onclick="app.showAddPositionModal()" 
                    class="bg-primary-500 text-white px-4 py-2 rounded hover:bg-primary-600 transition-colors">
              Add Position
            </button>
          </div>
          
          ${this.portfolio.positions.length === 0 ? `
            <div class="text-center py-8">
              <div class="text-white/60 mb-4">
                <svg class="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                </svg>
              </div>
              <p class="text-white/60">No positions yet</p>
              <p class="text-sm text-white/40 mt-2">Add your first position to get started</p>
            </div>
          ` : `
            <div class="space-y-3">
              ${this.portfolio.positions.map(position => this.renderPosition(position)).join('')}
            </div>
          `}
        </div>

        <!-- Portfolio Allocation Chart -->
        ${this.portfolio.positions.length > 0 ? `
          <div class="glass-effect rounded-xl p-6">
            <h3 class="text-xl font-semibold text-white mb-4">Portfolio Allocation</h3>
            <canvas id="allocation-chart" width="400" height="200"></canvas>
          </div>
        ` : ''}
      </div>
    `

    // Render allocation chart if positions exist
    if (this.portfolio.positions.length > 0) {
      this.renderAllocationChart()
    }
  }

  renderPosition(position) {
    const currentValue = position.shares * position.lastPrice
    const costBasis = position.shares * position.avgPrice
    const gainLoss = currentValue - costBasis
    const gainLossPercent = (gainLoss / costBasis) * 100

    return `
      <div class="bg-white/10 rounded-lg p-4">
        <div class="flex justify-between items-start mb-2">
          <div>
            <h4 class="font-semibold text-white">${position.symbol}</h4>
            <p class="text-sm text-white/60">${position.type}</p>
          </div>
          <div class="text-right">
            <div class="font-semibold text-white">$${currentValue.toFixed(2)}</div>
            <div class="text-sm ${gainLoss >= 0 ? 'text-green-400' : 'text-red-400'}">
              ${gainLoss >= 0 ? '+' : ''}${gainLoss.toFixed(2)} (${gainLossPercent.toFixed(2)}%)
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div class="text-white/60">Shares</div>
            <div class="text-white">${position.shares}</div>
          </div>
          <div>
            <div class="text-white/60">Avg Price</div>
            <div class="text-white">$${position.avgPrice.toFixed(2)}</div>
          </div>
          <div>
            <div class="text-white/60">Last Price</div>
            <div class="text-white">$${position.lastPrice.toFixed(2)}</div>
          </div>
        </div>
        
        <div class="flex justify-end mt-3 space-x-2">
          <button onclick="app.sellPosition('${position.symbol}')" 
                  class="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600 transition-colors">
            Sell
          </button>
          <button onclick="app.showChart('${position.symbol}', ${position.type === 'crypto'})" 
                  class="bg-primary-500 text-white px-3 py-1 rounded text-sm hover:bg-primary-600 transition-colors">
            Chart
          </button>
        </div>
      </div>
    `
  }

  renderAllocationChart() {
    // This would render a pie chart showing portfolio allocation
    // Implementation would depend on the charting library used
    console.log('Rendering allocation chart...')
  }

  async getRiskMetrics() {
    try {
      const positions = this.portfolio.positions.map(pos => ({
        symbol: pos.symbol,
        allocation: (pos.shares * pos.lastPrice / this.portfolio.totalValue) * 100
      }))

      const response = await this.api.getRiskAnalysis(this.portfolio.totalValue, positions)
      return response.success ? response.riskMetrics : null
    } catch (error) {
      console.error('Error getting risk metrics:', error)
      return null
    }
  }

  exportPortfolio() {
    const data = {
      ...this.portfolio,
      exportDate: new Date().toISOString(),
      summary: this.getPortfolioSummary()
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `portfolio-${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  importPortfolio(file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result)
        this.portfolio = {
          totalValue: data.totalValue || 10000,
          positions: data.positions || [],
          cash: data.cash || 10000
        }
        this.savePortfolio()
        // Refresh the display
        const container = document.getElementById('portfolio-container')
        if (container) {
          this.renderPortfolio(container)
        }
      } catch (error) {
        console.error('Error importing portfolio:', error)
        alert('Error importing portfolio file')
      }
    }
    reader.readAsText(file)
  }
}