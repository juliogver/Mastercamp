<template>
  <div>
    <header class="bg-white dark:bg-gray-900">
      <!-- Header code goes here -->
    </header>

    <section class="bg-gray-900 text-white">
      <div class="mx-auto max-w-screen-xl px-4 py-32 lg:flex lg:h-screen lg:items-center">
        <div class="mx-auto max-w-3xl text-center">
          <h1 class="bg-gradient-to-r from-green-300 via-blue-500 to-purple-600 bg-clip-text text-3xl font-extrabold text-transparent sm:text-5xl">
            Analyze your dataset.
          </h1>
          <p class="mx-auto mt-4 max-w-xl sm:text-xl/relaxed">
            Thanks to our system, you'll be able to analyze the feelings from the comments of your dataset.
          </p>
          <div class="mt-8 flex flex-wrap justify-center gap-4">
            <input type="file" ref="csvFileInput" accept=".csv" class="hidden">
            <button
              class="block w-full rounded border border-blue-600 bg-blue-600 px-12 py-3 text-sm font-medium text-white hover:bg-transparent hover:text-white focus:outline-none focus:ring active:text-opacity-75 sm:w-auto"
              @click="uploadCSV"
            >
              Upload CSV
            </button>
            <button
              class="block w-full rounded border border-blue-600 px-12 py-3 text-sm font-medium text-white hover:bg-blue-600 focus:outline-none focus:ring active:bg-blue-500 sm:w-auto"
              @click="analyzeComments"
            >
              Analyze Comments
            </button>
          </div>
          <div id="chartContainer" class="mt-8 mx-auto" style="max-width: 600px; height: 400px;"></div>
        </div>
      </div>
    </section>

    <footer class="bg-gray-100 dark:bg-gray-900">
      <!-- Footer code goes here -->
    </footer>
  </div>
</template>

<script>
import ApexCharts from 'apexcharts';

export default {
  mounted() {
    this.renderChart();
  },
  methods: {
    renderChart() {
      const sentimentData = this.analyzeSentiment([]);
      this.generatePlot(sentimentData);
    },
    uploadCSV() {
      this.$refs.csvFileInput.click();
    },
    analyzeComments() {
      const fileInput = this.$refs.csvFileInput;
      const file = fileInput.files[0];

      if (file) {
        const reader = new FileReader();

        reader.onload = (e) => {
          const contents = e.target.result;
          const comments = this.parseCSV(contents);
          const sentimentData = this.analyzeSentiment(comments);
          this.generatePlot(sentimentData);
        };

        reader.readAsText(file);
      }
    },
    parseCSV(csvContent) {
      const lines = csvContent.split('\n');
      const comments = [];

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line !== '') {
          const comment = line.split(',')[1].trim();
          comments.push(comment);
        }
      }

      return comments;
    },
    analyzeSentiment() {
      // Placeholder function for sentiment analysis
      const sentimentData = [
        { sentiment: 'Positive', count: 10 },
        { sentiment: 'Neutral', count: 5 },
        { sentiment: 'Negative', count: 3 }
      ];

      return sentimentData;
    },
    generatePlot(sentimentData) {
      const chartOptions = {
        series: sentimentData.map((data) => data.count),
        labels: sentimentData.map((data) => data.sentiment),
        chart: { type: 'pie' },
        colors: ['#7EE3B4', '#4673AA', '#8A3BEB'],
      };

      const chart = new ApexCharts(document.getElementById('chartContainer'), chartOptions);
      chart.render();
    },
  },
};
</script>

<style>
/* Add any custom styles here */
</style>
