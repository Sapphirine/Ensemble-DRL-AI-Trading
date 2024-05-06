import './App.css';
import perf_ov from './perf_overview.json'

import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const LineGraph = () => {
  const svgRef = useRef(null);

  useEffect(() => {
    // JSON data
    const jsonData = [
      { "date": 1451865600000, "averageaction": 1000000.0 },
      { "date": 1451952000000, "averageaction": 999947.944525049 },
      { "date": 1452038400000, "averageaction": 998561.9598573199 }
    ];

    // Parse dates
    const parseDate = d3.timeParse('%Q');
    const data = perf_ov.map(d => ({
      date: parseDate(d.date),
      averageAction: d.averageaction
    }));

    // Set dimensions and margins
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };

    // Create SVG element
    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Set X scale
    const x = d3.scaleTime()
      .domain(d3.extent(data, d => d.date))
      .range([0, width]);

    // Set Y scale
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.averageAction)])
      .nice()
      .range([height, 0]);

    // Create line generator
    const line = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.averageAction));

    // Append line to SVG
    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 1.5)
      .attr('d', line);

    // Append X axis to SVG
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    // Append Y axis to SVG
    svg.append('g')
      .call(d3.axisLeft(y));

  }, []);

  return (        
      <div className="App">
        <header className="App-header">
        <h1>Algorithm Performance:</h1>
        <svg ref={svgRef}></svg>
        </header>
      </div>) 
  
  
 
};

export default LineGraph;

