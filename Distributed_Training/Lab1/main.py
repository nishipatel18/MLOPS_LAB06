import ray
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import titanic

PER_WORKER_N_ESTIMATORS = 50
NUM_WORKERS = 2

if ray.is_initialized():
    ray.shutdown()
ray.init()

X_train, X_test, y_train, y_test = titanic.titanic_dataset()
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

X_train_ref = ray.put(X_train_s)
X_test_ref  = ray.put(X_test_s)
y_train_ref = ray.put(y_train)
y_test_ref  = ray.put(y_test)

@ray.remote
def train_worker(X_train, X_test, y_train, y_test, n_estimators, worker_id):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=worker_id)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Worker {worker_id}: n_estimators={n_estimators}, accuracy={acc:.4f}")
    return worker_id, acc

global_n_estimators = PER_WORKER_N_ESTIMATORS * NUM_WORKERS
results_ref = [
    train_worker.remote(X_train_ref, X_test_ref, y_train_ref, y_test_ref,
                        PER_WORKER_N_ESTIMATORS, i)
    for i in range(NUM_WORKERS)
]
results = ray.get(results_ref)
for worker_id, acc in results:
    print(f"Worker {worker_id} accuracy: {acc:.4f}")
ray.shutdown()
