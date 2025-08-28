import Chart from 'chart.js/auto'
import 'chartjs-adapter-date-fns'

export class ChartManager {
  constructor() {
    this.charts = {}
  }

  createChart(canvasId, data, symbol) {
    const canvas = document.getElementById(canvasId)
    if (!canvas) return

    // Destroy existing chart
    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy()
    }

    const ctx = canvas.getContext('2d')
    
    // Prepare data for Chart.js
    const labels = data.dates
    const prices = data.prices.close
    const volumes = data.volume
    const sma20 = data.indicators.sma20
    const sma50 = data.indicators.sma50
    const rsi = data.indicators.rsi

    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Close Price',
            data: prices,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
          },
          {
            label: 'SMA 20',
            data: sma20,
            borderColor: 'rgb(249, 115, 22)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'SMA 50',
            data: sma50,
            borderColor: 'rgb(34, 197, 94)',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          title: {
            display: true,
            text: `${symbol} Price Chart`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            display: true,
            position: 'top'
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: 'white',
            bodyColor: 'white',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                const label = context.dataset.label || ''
                const value = context.parsed.y
                return `${label}: $${value.toFixed(2)}`
              }
            }
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM dd'
              }
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.1)'
            }
          },
          y: {
            beginAtZero: false,
            grid: {
              color: 'rgba(0, 0, 0, 0.1)'
            },
            ticks: {
              callback: function(value) {
                return '$' + value.toFixed(2)
              }
            }
          }
        },
        elements: {
          point: {
            radius: 0,
            hoverRadius: 4
          }
        }
      }
    })

    return this.charts[canvasId]
  }

  createCandlestickChart(canvasId, data, symbol) {
    // For candlestick charts, we would need a different library like Chart.js with candlestick plugin
    // or use Plotly.js. For now, we'll use the line chart
    return this.createChart(canvasId, data, symbol)
  }

  createVolumeChart(canvasId, data, symbol) {
    const canvas = document.getElementById(canvasId)
    if (!canvas) return

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy()
    }

    const ctx = canvas.getContext('2d')
    
    this.charts[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.dates,
        datasets: [{
          label: 'Volume',
          data: data.volume,
          backgroundColor: 'rgba(99, 102, 241, 0.6)',
          borderColor: 'rgb(99, 102, 241)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `${symbol} Volume`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM dd'
              }
            }
          },
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return (value / 1000000).toFixed(1) + 'M'
              }
            }
          }
        }
      }
    })

    return this.charts[canvasId]
  }

  createRSIChart(canvasId, data, symbol) {
    const canvas = document.getElementById(canvasId)
    if (!canvas) return

    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy()
    }

    const ctx = canvas.getContext('2d')
    
    this.charts[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [{
          label: 'RSI',
          data: data.indicators.rsi,
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: `${symbol} RSI`,
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          legend: {
            display: false
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM dd'
              }
            }
          },
          y: {
            min: 0,
            max: 100,
            ticks: {
              stepSize: 20
            }
          }
        },
        plugins: {
          annotation: {
            annotations: {
              overbought: {
                type: 'line',
                yMin: 70,
                yMax: 70,
                borderColor: 'red',
                borderWidth: 2,
                borderDash: [5, 5]
              },
              oversold: {
                type: 'line',
                yMin: 30,
                yMax: 30,
                borderColor: 'green',
                borderWidth: 2,
                borderDash: [5, 5]
              }
            }
          }
        }
      }
    })

    return this.charts[canvasId]
  }

  destroyChart(canvasId) {
    if (this.charts[canvasId]) {
      this.charts[canvasId].destroy()
      delete this.charts[canvasId]
    }
  }

  destroyAllCharts() {
    Object.keys(this.charts).forEach(canvasId => {
      this.destroyChart(canvasId)
    })
  }
}