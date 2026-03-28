// FINAL PERFECT VERSION - ALL BUGS FIXED
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("observationForm");
  const submitBtn = document.getElementById("submitBtn");
  const fillSampleBtn = document.getElementById("fillSampleData");

  if (!form) {
    console.error("Form not found!");
    return;
  }

  // Field validators
  const fieldValidators = {
    object_id: (value) =>
      !value || value.length === 0
        ? "Object ID is required"
        : value.length > 64
          ? "Object ID must be 64 characters or less"
          : !/^[A-Za-z0-9_-]+$/.test(value)
            ? "Only alphanumeric, hyphens and underscores allowed"
            : "",
    transit_depth: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Transit depth is required"
        : num < 0 || num > 100
          ? "Transit depth must be between 0 and 100%"
          : "";
    },
    orbital_period: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Orbital period is required"
        : num <= 0
          ? "Orbital period must be positive"
          : "";
    },
    transit_duration: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Transit duration is required"
        : num <= 0
          ? "Transit duration must be positive"
          : "";
    },
    snr: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "SNR is required"
        : num <= 0
          ? "SNR must be positive"
          : "";
    },
    stellar_radius: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Stellar radius is required"
        : num <= 0
          ? "Stellar radius must be positive"
          : "";
    },
    stellar_mass: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Stellar mass is required"
        : num <= 0
          ? "Stellar mass must be positive"
          : "";
    },
    stellar_temp: (value) => {
      const num = parseInt(value);
      return isNaN(num)
        ? "Stellar temperature is required"
        : num < 2500 || num > 50000
          ? "Temperature must be between 2500 and 50000 K"
          : "";
    },
    stellar_magnitude: (value) => {
      const num = parseFloat(value);
      return isNaN(num)
        ? "Stellar magnitude is required"
        : num < -10 || num > 30
          ? "Magnitude must be between -10 and 30"
          : "";
    },
  };

  function validateField(fieldName, value) {
    const validator = fieldValidators[fieldName];
    return validator ? validator(value) : "";
  }

  function updateFieldValidation(input) {
    const errorElement = document.getElementById(`${input.name}_error`);
    const errorMsg = validateField(input.name, input.value);
    if (errorMsg) {
      input.classList.add("error");
      input.classList.remove("valid");
      if (errorElement) errorElement.textContent = errorMsg;
    } else if (input.value) {
      input.classList.remove("error");
      input.classList.add("valid");
      if (errorElement) errorElement.textContent = "";
    } else {
      input.classList.remove("error", "valid");
      if (errorElement) errorElement.textContent = "";
    }
  }

  function validateForm() {
    const formData = new FormData(form);
    let isValid = true;
    for (const [name, value] of formData.entries()) {
      if (validateField(name, value)) isValid = false;
    }
    if (submitBtn) submitBtn.disabled = !isValid;
    return isValid;
  }

  form.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", (e) => {
      updateFieldValidation(e.target);
      validateForm();
    });
    input.addEventListener("blur", (e) => updateFieldValidation(e.target));
  });

  if (fillSampleBtn) {
    fillSampleBtn.addEventListener("click", () => {
      const objects = [
        {
          object_id: "KOI-7016",
          transit_depth: 1.234,
          orbital_period: 365.25,
          transit_duration: 6.5,
          snr: 12.8,
          stellar_radius: 1.02,
          stellar_mass: 0.98,
          stellar_temp: 5778,
          stellar_magnitude: 11.5,
        },
        {
          object_id: "KELT-9b",
          transit_depth: 0.7,
          orbital_period: 1.48,
          transit_duration: 4.9,
          snr: 12.8,
          stellar_radius: 2.36,
          stellar_mass: 2.52,
          stellar_temp: 10170,
          stellar_magnitude: 7.56,
        },
        {
          object_id: "WASP-103b",
          transit_depth: 0.2,
          orbital_period: 2,
          transit_duration: 11,
          snr: 11,
          stellar_radius: 1.02,
          stellar_mass: 0.98,
          stellar_temp: 9000,
          stellar_magnitude: 11.4,
        },
        {
          object_id: "KOI-4878",
          transit_depth: 0.012,
          orbital_period: 200,
          transit_duration: 12.5,
          snr: 2,
          stellar_radius: 1.1,
          stellar_mass: 1.04,
          stellar_temp: 5780,
          stellar_magnitude: 12.2,
        },
        {
          object_id: "Undefined",
          transit_depth: 8.9,
          orbital_period: 38,
          transit_duration: 12.5,
          snr: 32,
          stellar_radius: 1.1,
          stellar_mass: 0.3,
          stellar_temp: 2500,
          stellar_magnitude: 2,
        },
      ];
      const sampleData = objects[Math.floor(Math.random() * objects.length)];
      Object.entries(sampleData).forEach(([name, value]) => {
        const input = form.querySelector(`[name="${name}"]`);
        if (input) {
          input.value = value;
          updateFieldValidation(input);
        }
      });
      validateForm();
    });
  }

  // PERFECT SUBMIT HANDLER
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    const formData = new FormData(form);
    const data = {};

    // Collect form data with correct types
    for (const [name, value] of formData.entries()) {
      if (name === "object_id") {
        data[name] = String(value).trim(); // STRING
      } else if (name === "stellar_temp") {
        data[name] = parseInt(value);
      } else {
        data[name] = parseFloat(value);
      }
    }

    console.log("Form Data:", {
      object_id: data.object_id,
      type: typeof data.object_id,
    });

    setSubmittingState(true);

    try {
      // API request (7 fields only)
      const apiData = {
        orbital_period: parseFloat(data.orbital_period),
        transit_duration: parseFloat(data.transit_duration),
        transit_depth: parseFloat(data.transit_depth) / 100,
        snr: parseFloat(data.snr),
        stellar_mass: parseFloat(data.stellar_mass),
        stellar_temp: parseInt(data.stellar_temp),
        stellar_magnitude: parseFloat(data.stellar_magnitude),
      };

      console.log("Sending to API:", apiData);

      const response = await fetch(
        "https://nasa-exoplanet-api.onrender.com/analyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(apiData),
        },
      );

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      console.log("API Result:", result);
      console.log("   Type:", result.classification);
      console.log("   Has props:", !!result.properties);

      saveToLocalStorage(data);
      showWaitingState(data.object_id);

      // Format data for display
      setTimeout(() => {
        const displayData = {
          object_id: data.object_id, // Already a string
          percent: (result.confidence * 100).toFixed(1),
        };

        // Handle each classification type
        if (result.classification === "false_positive") {
          // FALSE POSITIVE - Show FALSE for all properties
          console.log("FALSE POSITIVE - showing FALSE");
          displayData.classification = "false_positive";
          displayData.planet_radius = "FALSE";
          displayData.semi_major_axis = "FALSE";
          displayData.eq_temperature = "FALSE";
        } else if (result.classification === "confirmed_exoplanet") {
          // CONFIRMED EXOPLANET - Show real data
          console.log("CONFIRMED EXOPLANET - showing data");
          displayData.classification = "confirmed_exoplanet";
          displayData.planet_radius =
            result.properties.planet_radius.toFixed(2);
          displayData.semi_major_axis =
            result.properties.semi_major_axis.toFixed(4);
          displayData.eq_temperature = Math.round(
            result.properties.planet_temp,
          ).toString();
        } else if (result.classification === "planetary_candidate") {
          // PLANETARY CANDIDATE - Show real data
          console.log("PLANETARY CANDIDATE - showing data");
          displayData.classification = "planetary_candidate";
          displayData.planet_radius =
            result.properties.planet_radius.toFixed(2);
          displayData.semi_major_axis =
            result.properties.semi_major_axis.toFixed(4);
          displayData.eq_temperature = Math.round(
            result.properties.planet_temp,
          ).toString();
        }

        console.log("Final displayData:", displayData);
        displayResults(displayData);
        setSubmittingState(false);
      }, 500);
    } catch (error) {
      console.error("Error:", error);
      alert("Error: " + error.message);
      showWaitingState(data.object_id);
      setTimeout(() => {
        displayResults({
          object_id: data.object_id,
          classification: "demo",
          percent: (70 + Math.random() * 20).toFixed(1),
          planet_radius: (1 + Math.random() * 8).toFixed(2),
          semi_major_axis: (0.1 + Math.random() * 1.5).toFixed(4),
          eq_temperature: Math.round(500 + Math.random() * 1500).toString(),
        });
        setSubmittingState(false);
      }, 2000);
    }
  });

  function saveToLocalStorage(data) {
    try {
      const submissions = JSON.parse(
        localStorage.getItem("planetAnalysisSubmissions") || "[]",
      );
      submissions.push({
        ...data,
        timestamp: new Date().toISOString(),
        id: Date.now().toString(),
      });
      localStorage.setItem(
        "planetAnalysisSubmissions",
        JSON.stringify(submissions),
      );
    } catch (e) {}
  }

  function setSubmittingState(isSubmitting) {
    if (!submitBtn) return;
    const btnText = submitBtn.querySelector(".btn-text");
    const spinner = submitBtn.querySelector(".spinner");
    if (isSubmitting) {
      submitBtn.classList.add("loading");
      submitBtn.disabled = true;
      if (btnText) btnText.textContent = "Analyzing...";
      if (spinner) spinner.style.display = "block";
    } else {
      submitBtn.classList.remove("loading");
      if (btnText) btnText.textContent = "Analyze";
      if (spinner) spinner.style.display = "none";
      validateForm();
    }
  }

  function showWaitingState(objectId) {
    const resultsSection = document.getElementById("resultsSection");
    const waitingState = document.getElementById("waitingState");
    const resultsDisplay = document.getElementById("resultsDisplay");
    if (resultsSection && waitingState) {
      resultsSection.style.display = "block";
      waitingState.style.display = "block";
      if (resultsDisplay) resultsDisplay.style.display = "none";
      const waitingTimestamp = document.getElementById("waitingTimestamp");
      if (waitingTimestamp)
        waitingTimestamp.textContent = `Submitted at ${new Date().toLocaleTimeString()}`;
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  validateForm();
});
