# Scaling Pawikan Sentinel: A Hub-and-Spoke Approach

This document outlines a strategy for scaling the Pawikan Sentinel project from a single-device setup to a robust, cost-effective system capable of monitoring an entire coastline. The recommended approach is to adopt a **Hub-and-Spoke** architecture.

## What is a Hub-and-Spoke Architecture?

A hub-and-spoke architecture is a model for distributed systems where a central "hub" connects to multiple "spokes."

*   **The Hub:** A central server that acts as the primary point for data collection, storage, and management.
*   **The Spokes:** Remote, independent nodes that perform tasks locally and communicate with the hub.

Think of it like an airport. The main airport is the hub, and the smaller, regional airports are the spokes. It's more efficient for the spokes to handle local traffic and only send long-distance passengers (or in our case, important data) to the hub for coordination.

**In the context of Pawikan Sentinel:**
*   **Spokes:** Low-cost, solar-powered inference devices deployed along the coast. They run the ML model to detect turtles.
*   **Hub:** A central server that receives detection alerts from all spokes, stores the data, and hosts the web dashboard for the whole operation.

### Example Project: Smart Agriculture

A common example is a smart agriculture system.
*   **Spokes:** In-field sensors that measure soil moisture, temperature, and nutrient levels. They operate independently on battery power.
*   **Hub:** A central server that collects data from all sensors. It runs analytics to provide farmers with a dashboard showing the health of their entire farm and can trigger automated actions, like turning on irrigation for a specific field.

---

## Building the Pawikan Sentinel Hub-and-Spoke System

Here is a breakdown of the components and recommended hardware for building out this architecture.

### The "Spoke": Edge Inference Device

The spoke is the workhorse in the field. Its only job is to run the ML model locally and send a small alert to the hub when it detects a turtle.

#### Hardware Options

| Option                  | Est. Cost | Pros                                       | Cons                               | Best For                               |
| ----------------------- | --------- | ------------------------------------------ | ---------------------------------- | -------------------------------------- |
| **Raspberry Pi 4B**     | ~$55      | Low cost, low power, you know it well      | Lower ML performance               | Getting started without new hardware   |
| **Raspberry Pi 5**      | ~$80      | Significant performance boost over Pi 4    | Higher power consumption           | A good balance of cost & performance   |
| **NVIDIA Jetson Nano**  | ~$150     | Dedicated AI hardware for best performance | Higher cost, more complex setup    | High-accuracy needs or future upgrades |

#### Software Components

*   **OS:** Raspberry Pi OS (or other lightweight Linux).
*   **Inference:** A Python script using an efficient runtime like **TensorFlow Lite** or **ONNX Runtime**.
*   **Communication:** An MQTT client (e.g., the `paho-mqtt` Python library) to publish detection events.

---

### The "Hub": Central Server

The hub is the central brain. It aggregates data from all spokes, manages the database, and provides the web dashboard.

#### Hosting Options

| Option                      | Est. Cost          | Pros                                           | Cons                                       | Best For                               |
| --------------------------- | ------------------ | ---------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| **Self-Hosted (Mini PC)**   | ~$150+ (one-time)  | Full control, no monthly fees                  | You are responsible for all maintenance    | Technical users who prefer a CAPEX model |
| **Virtual Private Server (VPS)** | **~$5-10/month**   | **Cost-effective, flexible, easy to scale**    | Requires server management                 | **The most recommended path**          |
| **Cloud "Free Tiers"**      | Free               | Great for testing and proof-of-concept         | Limited resources, may be outgrown quickly | Getting started with zero cost         |

#### Software Components

*   **OS:** A standard Linux distribution like Ubuntu.
*   **Database:** **PostgreSQL**. It's robust, open-source, and can handle data from hundreds of spokes.
*   **Communication:** An MQTT broker like **Mosquitto** to receive messages from the spokes.
*   **Backend:** Your existing FastAPI application, modified to subscribe to the MQTT broker.

---

## Recommended Path Forward

You can transition to this architecture incrementally:

1.  **Start Small:** Set up a **low-cost VPS** (e.g., from DigitalOcean or Hetzner) as your hub. Install PostgreSQL and Mosquitto on it.
2.  **Convert a Spoke:** Re-configure one of your existing **Raspberry Pi 4B** devices to be your first spoke. Update its software to run inference locally and send MQTT messages to your new hub.
3.  **Test and Validate:** Run this single-spoke, single-hub setup to ensure the entire pipeline works as expected.
4.  **Scale Incrementally:** As you get funding, deploy new spokes along the coast. If your hub gets overloaded, you can easily upgrade your VPS plan.

This approach minimizes upfront cost and risk while putting you on a clear path to a scalable, long-term solution for safeguarding sea turtles.
