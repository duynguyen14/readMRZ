/*
    Add source TransactionEVisa snapshot and field-by-field comparison columns
    to passport evaluation items. Safe to run repeatedly.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF OBJECT_ID(N'dbo.readmrz_passport_evaluation_items', N'U') IS NULL
    THROW 50001, 'Missing dbo.readmrz_passport_evaluation_items. Run create_passport_evaluation_tables.sql first.', 1;
GO

IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMatchStatus') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMatchStatus NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMatchReason') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMatchReason NVARCHAR(500) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceCandidateCount') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceCandidateCount INT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceTransactionEVisaId') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceTransactionEVisaId BIGINT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceTransactionEVisaGuid') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceTransactionEVisaGuid VARCHAR(50) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceCreatedDate') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceCreatedDate DATETIME2(3) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceModifiedDate') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceModifiedDate DATETIME2(3) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceFullPassportImage') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceFullPassportImage NVARCHAR(MAX) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceFaceImage') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceFaceImage NVARCHAR(MAX) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceFileSha256') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceFileSha256 CHAR(64) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMrzLineOne') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMrzLineOne NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMrzLineTwo') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMrzLineTwo NVARCHAR(128) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMrzLineOnePoint') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMrzLineOnePoint FLOAT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceMrzLineTwoPoint') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceMrzLineTwoPoint FLOAT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourcePassportType') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourcePassportType NVARCHAR(20) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceDatabasePassportType') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceDatabasePassportType NVARCHAR(50) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceIssuingCountry') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceIssuingCountry NVARCHAR(20) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceSurname') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceSurname NVARCHAR(255) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceGivenNames') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceGivenNames NVARCHAR(255) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceFullName') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceFullName NVARCHAR(512) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceDatabaseFullName') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceDatabaseFullName NVARCHAR(512) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourcePassportNumber') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourcePassportNumber NVARCHAR(50) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceNationality') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceNationality NVARCHAR(20) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceNationalityOriginal') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceNationalityOriginal NVARCHAR(50) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceDateOfBirth') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceDateOfBirth DATE NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceSex') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceSex NVARCHAR(20) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceDateOfExpiry') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceDateOfExpiry DATE NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceIssuePassportDate') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceIssuePassportDate DATE NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceIssuePassportPlace') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceIssuePassportPlace NVARCHAR(255) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourcePersonalNumber') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourcePersonalNumber NVARCHAR(100) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourcePassportImageNote') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourcePassportImageNote NVARCHAR(MAX) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceFaceImageNote') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceFaceImageNote NVARCHAR(MAX) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceDataJson') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceDataJson NVARCHAR(MAX) NULL;

IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'MrzLineOneMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD MrzLineOneMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'MrzLineTwoMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD MrzLineTwoMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'PassportTypeMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD PassportTypeMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'IssuingCountryMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD IssuingCountryMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SurnameMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SurnameMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'GivenNamesMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD GivenNamesMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'FullNameMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD FullNameMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'PassportNumberMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD PassportNumberMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'NationalityMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD NationalityMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'DateOfBirthMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD DateOfBirthMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SexMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SexMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'DateOfExpiryMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD DateOfExpiryMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'PersonalNumberMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD PersonalNumberMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ComparedFieldCount') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ComparedFieldCount INT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'MatchedFieldCount') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD MatchedFieldCount INT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'AllComparedFieldsMatch') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD AllComparedFieldsMatch BIT NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ComparisonJson') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ComparisonJson NVARCHAR(MAX) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'SourceComparedAt') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD SourceComparedAt DATETIME2(3) NULL;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_evaluation_items_SourceMatchStatus'
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_evaluation_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_evaluation_items_SourceMatchStatus CHECK (
        SourceMatchStatus IS NULL OR SourceMatchStatus IN (N'matched', N'not_found', N'ambiguous', N'error')
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_evaluation_items_SourceDataJson'
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_evaluation_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_evaluation_items_SourceDataJson CHECK (
        SourceDataJson IS NULL OR ISJSON(SourceDataJson) = 1
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_evaluation_items_ComparisonJson'
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_evaluation_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_evaluation_items_ComparisonJson CHECK (
        ComparisonJson IS NULL OR ISJSON(ComparisonJson) = 1
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_evaluation_items_SourceMatch'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_evaluation_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_evaluation_items_SourceMatch
        ON dbo.readmrz_passport_evaluation_items (RunId, SourceMatchStatus, CountryCode)
        INCLUDE (SourceTransactionEVisaId, AllComparedFieldsMatch, ComparedFieldCount, MatchedFieldCount);
END;
GO

IF OBJECT_ID(N'dbo.vw_readmrz_passport_source_comparison_by_country', N'V') IS NULL
    EXEC(N'CREATE VIEW dbo.vw_readmrz_passport_source_comparison_by_country AS SELECT 1 AS Placeholder;');
GO

ALTER VIEW dbo.vw_readmrz_passport_source_comparison_by_country
AS
SELECT
    item.RunId,
    item.CountryCode,
    COUNT_BIG(*) AS TotalItems,
    SUM(CASE WHEN item.SourceMatchStatus = N'matched' THEN 1 ELSE 0 END) AS SourceMatchedItems,
    SUM(CASE WHEN item.SourceMatchStatus = N'not_found' THEN 1 ELSE 0 END) AS SourceNotFoundItems,
    SUM(CASE WHEN item.SourceMatchStatus = N'ambiguous' THEN 1 ELSE 0 END) AS SourceAmbiguousItems,
    SUM(CASE WHEN item.SourceMatchStatus = N'error' THEN 1 ELSE 0 END) AS SourceErrorItems,
    SUM(CASE WHEN item.AllComparedFieldsMatch = 1 THEN 1 ELSE 0 END) AS AllFieldsCorrectItems,
    SUM(CASE WHEN item.AllComparedFieldsMatch = 0 THEN 1 ELSE 0 END) AS AnyFieldIncorrectItems,
    SUM(CASE WHEN item.MrzLineOneMatch = 0 THEN 1 ELSE 0 END) AS MrzLineOneIncorrectItems,
    SUM(CASE WHEN item.MrzLineTwoMatch = 0 THEN 1 ELSE 0 END) AS MrzLineTwoIncorrectItems,
    SUM(CASE WHEN item.PassportNumberMatch = 0 THEN 1 ELSE 0 END) AS PassportNumberIncorrectItems,
    SUM(CASE WHEN item.NationalityMatch = 0 THEN 1 ELSE 0 END) AS NationalityIncorrectItems,
    SUM(CASE WHEN item.DateOfBirthMatch = 0 THEN 1 ELSE 0 END) AS DateOfBirthIncorrectItems,
    SUM(CASE WHEN item.SexMatch = 0 THEN 1 ELSE 0 END) AS SexIncorrectItems,
    SUM(CASE WHEN item.DateOfExpiryMatch = 0 THEN 1 ELSE 0 END) AS DateOfExpiryIncorrectItems,
    SUM(CASE WHEN item.FullNameMatch = 0 THEN 1 ELSE 0 END) AS FullNameIncorrectItems
FROM dbo.readmrz_passport_evaluation_items item
GROUP BY item.RunId, item.CountryCode;
GO

SELECT N'Passport source comparison schema is ready.' AS Result;
