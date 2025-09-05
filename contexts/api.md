### `api.md`

**Your Role**: API Designer

* **Auth**:
  * `POST /auth/login`
  * `POST /auth/logout`
  * `POST /auth/change-password`
* **Contacts**:
  * `GET /api/contacts`
  * `POST /api/contacts`
  * `PUT /api/contacts/{id}`
  * `DELETE /api/contacts/{id}`
* **Cameras**:
  * `GET /api/cameras`
  * `POST /api/cameras`
  * `PUT /api/cameras/{id}`
  * `DELETE /api/cameras/{id}`
* **Detections**:
  * `GET /api/detections` (filter by date, camera).
* **Analytics**:
  * `GET /api/analytics/counts`
  * `GET /api/analytics/timeline`
