(() => {
  console.log("[VarElim] Loading Variable Elimination implementation...");

  class Factor {
    constructor(variables, cardinalities, values) {
      this.variables = variables;  
      this.cardinalities = cardinalities;  
      this.values = values; 
    }

    getStrides() {
      const strides = {};
      let stride = 1;
      for (let i = this.variables.length - 1; i >= 0; i--) {
        const v = this.variables[i];
        strides[v] = stride;
        stride *= this.cardinalities[v];
      }
      return strides;
    }

    getValue(assignment) {
      const strides = this.getStrides();
      let index = 0;
      for (const v of this.variables) {
        index += assignment[v] * strides[v];
      }
      return this.values[index];
    }

    setValue(assignment, value) {
      const strides = this.getStrides();
      let index = 0;
      for (const v of this.variables) {
        index += assignment[v] * strides[v];
      }
      this.values[index] = value;
    }

    multiply(other) {
      const newVars = [...new Set([...this.variables, ...other.variables])];
      const newCards = {};
      for (const v of newVars) {
        newCards[v] = this.cardinalities[v] || other.cardinalities[v];
      }

      const size = newVars.reduce((prod, v) => prod * newCards[v], 1);
      const newValues = new Array(size).fill(0);

      const newFactor = new Factor(newVars, newCards, newValues);
      
      const assignment = {};
      for (const v of newVars) assignment[v] = 0;
      
      for (let i = 0; i < size; i++) {
        const val1 = this.getValue(assignment);
        const val2 = other.getValue(assignment);
        newFactor.setValue(assignment, val1 * val2);
        
        for (let j = newVars.length - 1; j >= 0; j--) {
          const v = newVars[j];
          assignment[v]++;
          if (assignment[v] < newCards[v]) break;
          assignment[v] = 0;
        }
      }

      return newFactor;
    }

    marginalize(variable) {
      if (!this.variables.includes(variable)) return this;

      const newVars = this.variables.filter(v => v !== variable);
      const newCards = {};
      for (const v of newVars) {
        newCards[v] = this.cardinalities[v];
      }

      const size = newVars.reduce((prod, v) => prod * newCards[v], 1);
      const newValues = new Array(size).fill(0);
      const newFactor = new Factor(newVars, newCards, newValues);

      const assignment = {};
      for (const v of this.variables) assignment[v] = 0;

      const totalSize = this.values.length;
      for (let i = 0; i < totalSize; i++) {
        const value = this.values[i];
        newFactor.setValue(assignment, newFactor.getValue(assignment) + value);
        
        for (let j = this.variables.length - 1; j >= 0; j--) {
          const v = this.variables[j];
          assignment[v]++;
          if (assignment[v] < this.cardinalities[v]) break;
          assignment[v] = 0;
        }
      }

      return newFactor;
    }

    reduce(variable, value) {
      if (!this.variables.includes(variable)) return this;

      const newVars = this.variables.filter(v => v !== variable);
      const newCards = {};
      for (const v of newVars) {
        newCards[v] = this.cardinalities[v];
      }

      const size = newVars.reduce((prod, v) => prod * newCards[v], 1);
      const newValues = new Array(size).fill(0);
      const newFactor = new Factor(newVars, newCards, newValues);

      const assignment = {};
      for (const v of this.variables) {
        assignment[v] = v === variable ? value : 0;
      }

      for (let i = 0; i < size; i++) {
        newFactor.values[i] = this.getValue(assignment);
        
        for (let j = this.variables.length - 1; j >= 0; j--) {
          const v = this.variables[j];
          if (v === variable) continue;
          assignment[v]++;
          if (assignment[v] < this.cardinalities[v]) break;
          assignment[v] = 0;
        }
      }

      return newFactor;
    }

    normalize() {
      const sum = this.values.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        for (let i = 0; i < this.values.length; i++) {
          this.values[i] /= sum;
        }
      }
      return this;
    }
  }

  function variableElimination(bn, queryVar, evidence) {
    console.log("[VarElim] Query:", queryVar, "Evidence:", evidence);
    
    const factors = [];
    const allVars = new Set();
    const cardinalities = {};

    const nodesData = {};
    for (const n of bn.nodes || []) {
      const nodeId = String(n.id || n.name || n.label);
      nodesData[nodeId] = n;
      const states = n.states || ['0', '1'];
      cardinalities[nodeId] = states.length;
      allVars.add(nodeId);
    }

    const cpts = bn.cpts || {};
    if (Object.keys(cpts).length === 0) {
      console.error("[VarElim] No CPTs found in network!");
      return null;
    }

    console.log("[VarElim] Building factors from CPTs for", allVars.size, "nodes");

    for (const nodeId of allVars) {
      const node = nodesData[nodeId];
      const cptData = cpts[nodeId];
      
      if (!cptData) {
        console.warn("[VarElim] No CPT for node:", nodeId);
        continue;
      }

      const states = node.states || cptData.states || ['0', '1'];
      const nStates = states.length;
      const parents = cptData.parents || [];
      const cpt = cptData.cpt || {};

      console.log(`[VarElim] Node ${nodeId}: ${nStates} states, ${parents.length} parents`);

      const factorVars = [nodeId, ...parents];
      const factorCards = {};
      for (const v of factorVars) {
        factorCards[v] = cardinalities[v];
      }

      const factorSize = factorVars.reduce((prod, v) => prod * factorCards[v], 1);
      const factorValues = new Array(factorSize);

      const assignment = {};
      for (const v of factorVars) assignment[v] = 0;

      for (let i = 0; i < factorSize; i++) {
        let prob;

        if (parents.length === 0) {
          const nodeStateIdx = assignment[nodeId];
          const nodeStateValue = states[nodeStateIdx];
          const rootProbs = cpt["__ROOT__"] || cpt["root"] || cpt[""] || {};
          prob = rootProbs[String(nodeStateValue)] || (1.0 / nStates);
        } else {
          const parentAssignment = parents.map((p, idx) => {
            const parentStates = nodesData[p]?.states || ['0', '1'];
            const parentStateIdx = assignment[p];
            const parentStateValue = parentStates[parentStateIdx];
            return `${p}=${parentStateValue}`;
          });
          
          const conditionKey = parentAssignment.join('|');
          const nodeStateIdx = assignment[nodeId];
          const nodeStateValue = states[nodeStateIdx];
          
          const condProbs = cpt[conditionKey];
          if (!condProbs) {
            console.warn(`[VarElim] Missing CPT entry for ${nodeId} | ${conditionKey}`);
            prob = 1.0 / nStates;
          } else {
            prob = condProbs[String(nodeStateValue)] || (1.0 / nStates);
          }
        }

        factorValues[i] = prob;

        for (let j = factorVars.length - 1; j >= 0; j--) {
          const v = factorVars[j];
          assignment[v]++;
          if (assignment[v] < factorCards[v]) break;
          assignment[v] = 0;
        }
      }

      factors.push(new Factor(factorVars, factorCards, factorValues));
    }

    console.log("[VarElim] Built", factors.length, "factors");

    for (const [evidenceVar, evidenceVal] of Object.entries(evidence)) {
      const states = nodesData[evidenceVar]?.states || ['0', '1'];
      const stateIdx = states.indexOf(String(evidenceVal));
      
      if (stateIdx < 0) {
        console.warn(`[VarElim] Evidence value ${evidenceVal} not found in states of ${evidenceVar}`);
        continue;
      }

      console.log(`[VarElim] Applying evidence: ${evidenceVar} = ${evidenceVal} (index ${stateIdx})`);

      for (let i = 0; i < factors.length; i++) {
        factors[i] = factors[i].reduce(evidenceVar, stateIdx);
      }
    }

    const hiddenVars = [...allVars].filter(v => v !== queryVar && !evidence[v]);
    
    console.log("[VarElim] Eliminating variables:", hiddenVars);

    for (const elimVar of hiddenVars) {
      const relevantFactors = [];
      const remainingFactors = [];
      
      for (const factor of factors) {
        if (factor.variables.includes(elimVar)) {
          relevantFactors.push(factor);
        } else {
          remainingFactors.push(factor);
        }
      }

      if (relevantFactors.length === 0) continue;

      let product = relevantFactors[0];
      for (let i = 1; i < relevantFactors.length; i++) {
        product = product.multiply(relevantFactors[i]);
      }

      const marginalized = product.marginalize(elimVar);

      factors.length = 0;
      factors.push(...remainingFactors, marginalized);
    }

    if (factors.length === 0) {
      console.error("[VarElim] No factors remaining!");
      return null;
    }

    let result = factors[0];
    for (let i = 1; i < factors.length; i++) {
      result = result.multiply(factors[i]);
    }

    result.normalize();

    const queryStates = nodesData[queryVar]?.states || ['0', '1'];
    const output = {};
    
    for (let i = 0; i < queryStates.length; i++) {
      const assignment = { [queryVar]: i };
      output[String(queryStates[i])] = result.getValue(assignment);
    }

    console.log("[VarElim] Result:", output);
  
    const probValues = Object.values(output);
    const uniqueValues = new Set(probValues.map(v => v.toFixed(6)));
    if (uniqueValues.size === 1) {
      console.warn("[VarElim] WARNING: Result is UNIFORM! This suggests a bug.");
    } else {
      console.log("[VarElim] Result is non-uniform");
    }

    return output;
  }

  window.performVariableElimination = variableElimination;
  console.log("[VarElim] Variable Elimination ready!");
})();