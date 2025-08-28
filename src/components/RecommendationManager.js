export class RecommendationManager {
  constructor(api) {
    this.api = api
    this.recommendations = []
  }

  async getRecommendations(symbols) {
    try {
      const response = await this.api.analyzeAssets(symbols)
      if (response.success) {
        this.recommendations = response.recommendations
        return this.recommendations
      } else {
        throw new Error(response.error || 'Failed to get recommendations')
      }
    } catch (error) {
      console.error('Error getting recommendations:', error)
      throw error
    }
  }

  formatRecommendation(recommendation) {
    const actionColors = {
      'STRONG BUY': 'text-green-700 bg-green-50',
      'BUY': 'text-green-600 bg-green-50',
      'HOLD': 'text-yellow-600 bg-yellow-50',
      'SELL': 'text-red-600 bg-red-50',
      'STRONG SELL': 'text-red-700 bg-red-50'
    }

    const actionColor = actionColors[recommendation.action] || 'text-gray-600 bg-gray-50'

    return `
      <div class="recommendation-card ${recommendation.action.toLowerCase().replace(' ', '-')} rounded-lg p-6 mb-4">
        <div class="flex justify-between items-start mb-4">
          <div>
            <h4 class="text-xl font-bold text-gray-800">${recommendation.symbol}</h4>
            <p class="text-gray-600">${recommendation.type}</p>
            <span class="inline-block px-3 py-1 rounded-full text-sm font-medium ${actionColor}">
              ${recommendation.action}
            </span>
          </div>
          <div class="text-right">
            <div class="text-2xl font-bold text-gray-800">$${recommendation.currentPrice.toFixed(2)}</div>
            <div class="text-lg font-semibold ${recommendation.expectedReturn >= 0 ? 'text-green-600' : 'text-red-600'}">
              ${recommendation.expectedReturn >= 0 ? '+' : ''}${recommendation.expectedReturn.toFixed(2)}%
            </div>
          </div>
        </div>
        
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div class="text-center p-3 bg-white/50 rounded-lg">
            <div class="text-sm text-gray-600 mb-1">Confidence Score</div>
            <div class="text-lg font-semibold">${recommendation.score}/100</div>
            <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div class="bg-primary-500 h-2 rounded-full" style="width: ${recommendation.score}%"></div>
            </div>
          </div>
          <div class="text-center p-3 bg-white/50 rounded-lg">
            <div class="text-sm text-gray-600 mb-1">RSI</div>
            <div class="text-lg font-semibold">${recommendation.rsi.toFixed(1)}</div>
            <div class="text-xs ${recommendation.rsi > 70 ? 'text-red-600' : recommendation.rsi < 30 ? 'text-green-600' : 'text-gray-600'}">
              ${recommendation.rsi > 70 ? 'Overbought' : recommendation.rsi < 30 ? 'Oversold' : 'Neutral'}
            </div>
          </div>
          <div class="text-center p-3 bg-white/50 rounded-lg">
            <div class="text-sm text-gray-600 mb-1">MACD Signal</div>
            <div class="text-lg font-semibold">${recommendation.macdSignal.toFixed(4)}</div>
            <div class="text-xs ${recommendation.macdSignal > 0 ? 'text-green-600' : 'text-red-600'}">
              ${recommendation.macdSignal > 0 ? 'Bullish' : 'Bearish'}
            </div>
          </div>
          <div class="text-center p-3 bg-white/50 rounded-lg">
            <div class="text-sm text-gray-600 mb-1">BB Position</div>
            <div class="text-lg font-semibold">${recommendation.bbPosition.toFixed(2)}</div>
            <div class="text-xs ${recommendation.bbPosition > 0.8 ? 'text-red-600' : recommendation.bbPosition < 0.2 ? 'text-green-600' : 'text-gray-600'}">
              ${recommendation.bbPosition > 0.8 ? 'Upper Band' : recommendation.bbPosition < 0.2 ? 'Lower Band' : 'Middle'}
            </div>
          </div>
        </div>
        
        <div class="flex justify-between items-center">
          <div class="flex space-x-2">
            <button onclick="app.showChart('${recommendation.symbol}', ${recommendation.type === 'Crypto'})" 
                    class="bg-primary-500 text-white px-4 py-2 rounded hover:bg-primary-600 transition-colors">
              📈 View Chart
            </button>
            <button onclick="app.showDetails('${recommendation.symbol}')" 
                    class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 transition-colors">
              📊 Details
            </button>
          </div>
          <div class="text-right">
            <div class="text-sm text-gray-600">Last Updated</div>
            <div class="text-sm font-medium">${new Date().toLocaleTimeString()}</div>
          </div>
        </div>
      </div>
    `
  }

  renderRecommendations(container, recommendations) {
    if (!recommendations || recommendations.length === 0) {
      container.innerHTML = `
        <div class="text-center py-8">
          <div class="text-gray-500 mb-4">
            <svg class="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
          </div>
          <p class="text-gray-500">No recommendations available</p>
          <p class="text-sm text-gray-400 mt-2">Add some symbols and run analysis to get started</p>
        </div>
      `
      return
    }

    // Sort recommendations by score (highest first)
    const sortedRecommendations = [...recommendations].sort((a, b) => b.score - a.score)

    container.innerHTML = `
      <div class="mb-6">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-xl font-semibold text-gray-800">Analysis Results</h3>
          <div class="text-sm text-gray-600">
            ${recommendations.length} symbol${recommendations.length !== 1 ? 's' : ''} analyzed
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div class="bg-green-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-green-600">
              ${sortedRecommendations.filter(r => r.action.includes('BUY')).length}
            </div>
            <div class="text-sm text-green-700">Buy Signals</div>
          </div>
          <div class="bg-yellow-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-yellow-600">
              ${sortedRecommendations.filter(r => r.action === 'HOLD').length}
            </div>
            <div class="text-sm text-yellow-700">Hold Signals</div>
          </div>
          <div class="bg-red-50 rounded-lg p-4 text-center">
            <div class="text-2xl font-bold text-red-600">
              ${sortedRecommendations.filter(r => r.action.includes('SELL')).length}
            </div>
            <div class="text-sm text-red-700">Sell Signals</div>
          </div>
        </div>
      </div>
      
      <div class="space-y-4">
        ${sortedRecommendations.map(rec => this.formatRecommendation(rec)).join('')}
      </div>
    `
  }

  getRecommendationSummary(recommendations) {
    if (!recommendations || recommendations.length === 0) {
      return {
        total: 0,
        buy: 0,
        hold: 0,
        sell: 0,
        avgScore: 0,
        avgReturn: 0
      }
    }

    const buy = recommendations.filter(r => r.action.includes('BUY')).length
    const hold = recommendations.filter(r => r.action === 'HOLD').length
    const sell = recommendations.filter(r => r.action.includes('SELL')).length
    const avgScore = recommendations.reduce((sum, r) => sum + r.score, 0) / recommendations.length
    const avgReturn = recommendations.reduce((sum, r) => sum + r.expectedReturn, 0) / recommendations.length

    return {
      total: recommendations.length,
      buy,
      hold,
      sell,
      avgScore: Math.round(avgScore),
      avgReturn: Math.round(avgReturn * 100) / 100
    }
  }
}