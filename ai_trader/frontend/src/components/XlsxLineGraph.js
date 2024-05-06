import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import * as xlsx from 'xlsx';

const XlsxLineGraph = ({ filePath }) => {
  const svgRef = useRef();

  useEffect(() => {
    // Load XLSX file
    const workbook = xlsx.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const data = xlsx.utils.sheet_to_json(workbook.Sheets[sheetName], { header: 1 });

    // Extract data and convert to appropriate format
    const dates = data[0].slice(1); // Assuming dates are in the first row starting from the second column
    const seriesData = [];
    for (let i = 1; i < data.length; i++) {
      const series = {
        name: data[i][0], // Assuming series names are in the first column
        values: data[i].slice(1).map(d => +d) // Assuming data starts from the second column
      };
      seriesData.push(series);
    }

    // Set up dimensions and margins
    const margin = { top: 20, right: 30, bottom: 30, left: 50 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Append SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Set up scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(dates, d => new Date(d)))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(seriesData, d => d3.max(d.values))])
      .range([height, 0]);

    // Set up line generator
    const line = d3.line()
      .x((d, i) => xScale(new Date(dates[i])))
      .y(d => yScale(d));

    // Append lines
    svg.selectAll('.line')
      .data(seriesData)
      .enter().append('path')
      .attr('class', 'line')
      .attr('fill', 'none')
      .attr('stroke', (_, i) => d3.schemeCategory10[i % 10]) // Color lines
      .attr('stroke-width', 1.5)
      .attr('d', d => line(d.values));

    // Append axes
    svg.append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%Y-%m-%d')));

    svg.append('g')
      .call(d3.axisLeft(yScale));
  }, [filePath]);

  return <svg ref={svgRef}></svg>;
};

export default XlsxLineGraph;
