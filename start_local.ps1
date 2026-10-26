<#
.SYNOPSIS
Start local services for Business Legal Assistant (Qdrant + Neo4j) and verify connectivity.

.DESCRIPTION
This script:
1. Checks if Docker is running and exits with instructions if not.
2. Starts (or reuses) a Qdrant container named 'qdrant_bla' with volume mount
   ${PWD}/qdrant_storage:/qdrant/storage and ports 6333/6334.
3. Reminds the user to start the Neo4j Desktop database manually.
4. Health-checks Qdrant (HTTP GET /collections) and Neo4j (bolt RETURN 1).
5. Prints clear PASS/FAIL for each check.

.NOTES
- Qdrant container name: qdrant_bla
- Volume mount: ${PWD}/qdrant_storage:/qdrant/storage
- Ports: 6333 (HTTP), 6334 (gRPC)
- Requires Neo4j Desktop to be running with a database started.
#>

# Check Docker is running
$dockerCheck = docker info > $null 2>&1
if $LASTEXITCODE -ne 0 {
    Write-Host "❌ Docker is not running or not accessible." -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker is running." -ForegroundColor Green

# Start (or reuse) Qdrant container
Write-Host "Starting Qdrant container..." -ForegroundColor Cyan
$containerExists = docker ps -a --filter "name=qdrant_bla" --format "{{.Names}}" | Select-String "qdrant_bla"
if $containerExists {
    Write-Host "♻️  Reusing existing Qdrant container..." -ForegroundColor Yellow
    docker start qdrant_bla > $null 2>&1
} else {
    Write-Host "📦 Creating new Qdrant container..." -ForegroundColor Yellow
    docker run -d --name qdrant_bla \
        -p 6333:6333 -p 6334:6334 \
        -v ${PWD}/qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest > $null 2>&1
    Start-Sleep -Seconds 3
}
Write-Host "✅ Qdrant container is running." -ForegroundColor Green

# Health-check Qdrant
Write-Host "Health-checking Qdrant..." -ForegroundColor Cyan
$qdrantHealth = try { (docker exec qdrant_bla curl -s http://localhost:6333/collections > $null 2>&1) } catch { $false }
if ($qdrantHealth) {
    Write-Host "✅ Qdrant health-check PASSED." -ForegroundColor Green
} else {
    Write-Host "❌ Qdrant health-check FAILED." -ForegroundColor Red
    Write-Host "   Ensure the Qdrant container is running: docker ps | grep qdrant_bla" -ForegroundColor Yellow
}

# Neo4j reminder and health-check
Write-Host "Reminder: Please start Neo4j Desktop and ensure a database is running." -ForegroundColor Yellow
Write-Host "  Open Neo4j Desktop → Select your database → Click 'Start'" -ForegroundColor Cyan

# Health-check Neo4j (bolt RETURN 1)
Write-Host "Health-checking Neo4j..." -ForegroundColor Cyan
$neo4jHealth = $false
try {
    # Use the neo4j python script to verify connectivity
    $pythonScript = @"
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(URI, auth=(USER, PASS))
with driver.session() as session:
    result = session.run("RETURN 1")
    record = result.single()
    print(f"Neo4j RETURN 1: {record[0]}")
driver.close()
"@
$result = python -c $pythonScript 2>&1
    if $LASTEXITCODE -eq 0 -and $result -match "RETURN 1" {
        $neo4jHealth = $true
    }
} catch {
    # ignored
}
if ($neo4jHealth) {
    Write-Host "✅ Neo4j health-check PASSED." -ForegroundColor Green
} else {
    Write-Host "❌ Neo4j health-check FAILED." -ForegroundColor Red
    Write-Host "   Ensure Neo4j Desktop is running and a database is started." -ForegroundColor Yellow
}

Write-Host "`nAll checks complete.`n" -ForegroundColor Green