export class MarketOverview {
  constructor(api) {
    this.api = api
    this.marketData = null
    this.updateInterval = null
  }

  async loadMarketData() {
    try {
      const response = await this.api.getMarketOverview()
      if (response.success) {
        this.marketData = response.data
        return this.marketData
      } else {
        throw new Error(response.error || 'Failed to load market data')
      }
    } catch (error) {
      console.error('Error loading market data:', error)
      throw error
    }
  }

  renderMarketOverview(container) {
    if (!this.marketData) {
      container.innerHTML = `
        <div class="text-center py-8">
          <div class="loading-spinner mx-auto mb-4"></div>
          <p class="text-white/80">Loading market data...</p>
        </div>
      `
      return
    }

    container.innerHTML = `
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Stocks Section -->
        <div class="glass-effect rounded-xl p-6">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-semibold text-white">Popular Stocks</h3>
            <div class="text-sm text-white/60">
              ${this.marketData.stocks.length} stocks
            </div>
          </div>
          <div class="space-y-3">
            ${this.marketData.stocks.map(stock => this.renderAssetCard(stock, 'stock')).join('')}
          </div>
        </div>

        <!-- Crypto Section -->
        <div class="glass-effect rounded-xl p-6">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-xl font-semibold text-white">Popular Cryptocurrencies</h3>
            <div class="text-sm text-white/60">
              ${this.marketData.crypto.length} cryptos
            </div>
          </div>
          <div class="space-y-3">
            ${this.marketData.crypto.map(crypto => this.renderAssetCard(crypto, 'crypto')).join('')}
          </div>
        </div>
      </div>

      <!-- Market Summary -->
      <div class="glass-effect rounded-xl p-6 mt-6">
        <h3 class="text-xl font-semibold text-white mb-4">Market Summary</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          ${this.renderMarketSummary()}
        </div>
      </div>
    `
  }

  renderAssetCard(asset, type) {
    const changeColor = asset.change >= 0 ? 'text-green-400' : 'text-red-400'
    const changeIcon = asset.change >= 0 ? '↗' : '↘'
    const typeColor = type === 'crypto' ? 'bg-orange-500' : 'bg-blue-500'

    return `
      <div class="flex justify-between items-center bg-white/10 rounded-lg p-3 hover:bg-white/20 transition-colors cursor-pointer"
           onclick="app.quickAnalyze('${asset.symbol}', '${type}')">
        <div class="flex items-center space-x-3">
          <div class="w-10 h-10 ${typeColor} rounded-full flex items-center justify-center text-white font-bold text-sm">
            ${asset.symbol.substring(0, 2)}
          </div>
          <div>
            <div class="font-semibold text-white">${asset.symbol}</div>
            <div class="text-sm text-white/60">${asset.name}</div>
          </div>
        </div>
        <div class="text-right">
          <div class="font-semibold text-white">$${asset.price.toFixed(2)}</div>
          <div class="text-sm ${changeColor} flex items-center">
            <span class="mr-1">${changeIcon}</span>
            ${Math.abs(asset.change).toFixed(2)}%
          </div>
        </div>
      </div>
    `
  }

  renderMarketSummary() {
    if (!this.marketData) return ''

    const stocksUp = this.marketData.stocks.filter(s => s.change > 0).length
    const stocksDown = this.marketData.stocks.filter(s => s.change < 0).length
    const cryptoUp = this.marketData.crypto.filter(c => c.change > 0).length
    const cryptoDown = this.marketData.crypto.filter(c => c.change < 0).length

    const avgStockChange = this.marketData.stocks.reduce((sum, s) => sum + s.change, 0) / this.marketData.stocks.length
    const avgCryptoChange = this.marketData.crypto.reduce((sum, c) => sum + c.change, 0) / this.marketData.crypto.length

    return `
      <div class="text-center">
        <div class="text-2xl font-bold text-green-400">${stocksUp}</div>
        <div class="text-sm text-white/80">Stocks Up</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold text-red-400">${stocksDown}</div>
        <div class="text-sm text-white/80">Stocks Down</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold ${avgStockChange >= 0 ? 'text-green-400' : 'text-red-400'}">
          ${avgStockChange >= 0 ? '+' : ''}${avgStockChange.toFixed(2)}%
        </div>
        <div class="text-sm text-white/80">Avg Stock Change</div>
      </div>
      <div class="text-center">
        <div class="text-2xl font-bold ${avgCryptoChange >= 0 ? 'text-green-400' : 'text-red-400'}">
          ${avgCryptoChange >= 0 ? '+' : ''}${avgCryptoChange.toFixed(2)}%
        </div>
        <div class="text-sm text-white/80">Avg Crypto Change</div>
      </div>
    `
  }

  startAutoUpdate(container, interval = 60000) {
    this.stopAutoUpdate()
    
    this.updateInterval = setInterval(async () => {
      try {
        await this.loadMarketData()
        this.renderMarketOverview(container)
      } catch (error) {
        console.error('Auto-update failed:', error)
      }
    }, interval)
  }

  stopAutoUpdate() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }

  getMarketSentiment() {
    if (!this.marketData) return 'neutral'

    const allAssets = [...this.marketData.stocks, ...this.marketData.crypto]
    const positiveCount = allAssets.filter(asset => asset.change > 0).length
    const totalCount = allAssets.length

    const positiveRatio = positiveCount / totalCount

    if (positiveRatio > 0.6) return 'bullish'
    if (positiveRatio < 0.4) return 'bearish'
    return 'neutral'
  }

  getTopPerformers(count = 5) {
    if (!this.marketData) return []

    const allAssets = [...this.marketData.stocks, ...this.marketData.crypto]
    return allAssets
      .sort((a, b) => b.change - a.change)
      .slice(0, count)
  }

  getWorstPerformers(count = 5) {
    if (!this.marketData) return []

    const allAssets = [...this.marketData.stocks, ...this.marketData.crypto]
    return allAssets
      .sort((a, b) => a.change - b.change)
      .slice(0, count)
  }
}