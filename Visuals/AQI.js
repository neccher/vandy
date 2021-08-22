function init() {
    // Grab a reference to the dropdown select element
    var selector = d3.select("#selDataset");
  
    // Use the list of sample names to populate the select options
    d3.json("samples.json").then((data) => {
      var sampleNames = data.names;
  
      sampleNames.forEach((sample) => {
        selector
          .append("option")
          .text(sample)
          .property("value", sample);
      });
  
      // Use the first sample from the list to build the initial plots
      var firstSample = sampleNames[0];
      buildCharts(firstSample);
      buildMetadata(firstSample);
    });
  }
  
  // Initialize the dashboard
  init();
  
  function optionChanged(newSample) {
    // Fetch new data each time a new sample is selected
    buildMetadata(newSample);
    buildCharts(newSample);
    
  }
  
  // Demographics Panel 
  function buildMetadata(sample) {
    d3.json("samples.json").then((data) => {
      var metadata = data.metadata;
      // Filter the data for the object with the desired sample number
      var resultArray = metadata.filter(sampleObj => sampleObj.id == sample);
      var result = resultArray[0];
      // Use d3 to select the panel with id of `#sample-metadata`
      var PANEL = d3.select("#sample-metadata");
  
      // Use `.html("") to clear any existing metadata
      PANEL.html("");
  
      // Use `Object.entries` to add each key and value pair to the panel
      // Hint: Inside the loop, you will need to use d3 to append new
      // tags for each key-value in the metadata.
      Object.entries(result).forEach(([key, value]) => {
        PANEL.append("h6").text(`${key.toUpperCase()}: ${value}`);
      });
  
    });
  }
  
  // -----BUILD CHART-----
  
  // 1. Create the buildCharts function.
  function buildCharts(sample) {
    // 2. Use d3.json to load and retrieve the samples.json file 
    d3.json("samples.json").then((data) => {
        console.log(data); 
  
      // 3. Create a variable that holds the samples array. 
      var sampleData = data.samples;
      // 4. Create a variable that filters the samples for the object with the desired sample number.
      var buildingArray = sampleData.filter(sampleObj => sampleObj.id == sample);
      var frequencyArray = data.metadata.filter(sampleObj => sampleObj.id == sample);
      //  5. Create a variable that holds the first sample in the array.
      var results = buildingArray[0];
      var freq_results = frequencyArray[0]; 
  
      // 6. Create variables that hold the otu_ids, otu_labels, and sample_values.
      var otu_ids = results.otu_ids;
      var otu_labels = results.otu_labels;
      var sample_values = results.sample_values;
      var washing_frequency = freq_results.wfreq;
      // 7. Create the yticks for the bar chart.
      // Hint: Get the the top 10 otu_ids and map them in descending order  
      //  so the otu_ids with the most bacteria are last. 
  
      var yticks = otu_ids.slice(0, 10).map(otuID => `OTU ${otuID}`).reverse();
  
  // ---BAR CHART---
  
      // 8. Create the trace for the bar chart. 
      var barData = [
        {
          y: yticks,
          x: sample_values.slice(0, 10).reverse(),
          text: otu_labels.slice(0, 10).reverse(),
          type: "bar",
          orientation: "h",
        }      
      ];
      // 9. Create the layout for the bar chart. 
      var barLayout = {
        title: "Top 10 Bacteria Cultures Found",
          margin: { t: 30, l: 150 }     
      };
      // 10. Use Plotly to plot the data with the layout. 
      Plotly.newPlot("bar", barData, barLayout);
    
  //  ---BUBBLE CHART---
  
      // 1. Create the trace for the bubble chart.
      var bubbleData = [
          {
            x: otu_ids,
            y: sample_values,
            text: otu_labels,
            mode: "markers",
            marker: {
              size: sample_values,
              color: otu_ids,
              colorscale: "Earth"
            }
          }   
        ];
  
      // 2. Create the layout for the bubble chart.
      var bubbleLayout = {
        title: "Bacteria Cultures Per Sample",
          hovermode: "closest",
          xaxis: { title: "OTU ID" },
      };
  
      // 3. Use Plotly to plot the data with the layout.
      Plotly.newPlot("bubble", bubbleData, bubbleLayout); 
  
  // ---GAUGE CHART---
    
      // 4. Create the trace for the gauge chart.
      var gaugeData = [
          {
            domain: { x: [0, 500], y: [0, 500] },
            value: washing_frequency,
            title: '<b>Air Quality Index Indicator</b> <br> Levels of Health Concern',
            type: "indicator",
            
            mode: "gauge+number",
            gauge: { 
                bar: { color: "Black"},
                axis: { range: [null, 500] },
                     steps: [
                        {range: [0, 50], color: "green"},
                        {range: [51, 100], color: "yellow"},
                        {range: [101, 150], color: "orange"},
                        {range: [151, 200], color: "red"},
                        {range: [201, 300], color: "purple"},
                        {range: [301, 500], color: "maroon"}
                    ]}
              
          }
      ];
      
      // 5. Create the layout for the gauge chart.
      var gaugeLayout = { 
              width: 700, 
              height: 600, 
              margin: { t: 20, b: 40, l:100, r:100 }     
      };
  
      // 6. Use Plotly to plot the gauge data and layout.
      Plotly.newPlot("gauge", gaugeData, gaugeLayout);
  
    });
  }